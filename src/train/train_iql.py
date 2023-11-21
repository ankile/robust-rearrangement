import argparse
import os
from pathlib import Path

import furniture_bench  # noqa: F401
import numpy as np
import torch
from diffusers.optimization import get_scheduler

# from ipdb import set_trace as bp
from ml_collections import ConfigDict
from src.common.earlystop import EarlyStopper
from src.common.pytorch_util import dict_apply
from src.data.dataloader import FixedStepsDataloader
from src.data.dataset import FurnitureQFeatureDataset
from src.data.normalizer import StateActionNormalizer
from src.eval import do_rollout_evaluation
from src.gym import get_env
from src.behavior.actor import ImplicitQActor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb


def main(config: ConfigDict):
    env = None
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    normalizer = StateActionNormalizer()

    if config.observation_type == "image":
        raise NotImplementedError("Image dataset not implemented yet")
    elif config.observation_type == "feature":
        dataset = FurnitureQFeatureDataset(
            dataset_path=config.datasim_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            normalizer=normalizer,
            normalize_features=config.vision_encoder.normalize_features,
            data_subset=config.data_subset,
        )
    else:
        raise ValueError(f"Unknown observation type: {config.observation_type}")

    # Split the dataset into train and test
    train_size = int(len(dataset) * (1 - config.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Update the config object with the action dimension
    config.action_dim = dataset.action_dim
    config.robot_state_dim = dataset.robot_state_dim

    # Create the policy network
    actor = ImplicitQActor(
        device=device,
        encoder_name=config.vision_encoder.model,
        freeze_encoder=config.vision_encoder.freeze,
        normalizer=normalizer,
        config=config,
    )

    # TODO: Remove this hacky way of doing things:
    # Warm-start only the actor from the following file:
    noise_net_wts_path = "/data/scratch/ankile/furniture-diffusion/models/actor_best.pt"
    # Allow missing keys as we are only loading the actor
    actor.load_state_dict(torch.load(noise_net_wts_path), strict=False)

    # Freeze the policy network for experiments
    if config.freeze_policy:
        for param in actor.model.parameters():
            param.requires_grad = False

    # AdamW optimizer for the actor
    parameter_groups = []

    if not config.freeze_policy:
        parameter_groups.append(
            {
                "params": actor.model.parameters(),
                "lr": config.actor_lr,
                "weight_decay": config.weight_decay,
            }
        )
    parameter_groups.append(
        {
            "params": actor.q_network.parameters(),
            "lr": config.critic_lr,
            "weight_decay": config.critic_weight_decay,
        }
    )
    parameter_groups.append(
        {
            "params": actor.value_network.parameters(),
            "lr": config.critic_lr,
            "weight_decay": config.critic_weight_decay,
        }
    )
    optimizer = torch.optim.AdamW(parameter_groups)

    if config.load_checkpoint_path is not None:
        print(f"Loading checkpoint from {config.load_checkpoint_path}")
        actor.load_state_dict(torch.load(config.load_checkpoint_path))

    # Update the config object with the observation dimension
    config.obs_dim = actor.obs_dim

    # Create dataloaders
    trainloader = FixedStepsDataloader(
        dataset=train_dataset,
        n_batches=config.steps_per_epoch,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    n_batches = len(trainloader)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * config.num_epochs,
    )

    early_stopper = EarlyStopper(
        patience=config.early_stopper.patience,
        smooth_factor=config.early_stopper.smooth_factor,
    )

    # Init wandb (move it down so that we can fail before starting a run in case we fail)
    wandb.init(
        project="iql-offline-comparison",
        entity="robot-rearrangement",
        config=config.to_dict(),
        mode="online" if not config.dryrun else "disabled",
        notes="Run baseline: Don't use IQL.",
    )

    # Watch the model
    wandb.watch(actor, log="all")

    # Save stats to wandb and update the config object
    wandb.log(
        {
            "num_samples": len(train_dataset),
            "num_samples_test": len(test_dataset),
            "num_episodes": int(len(dataset.episode_ends) * (1 - config.test_split)),
            "num_episodes_test": int(len(dataset.episode_ends) * config.test_split),
            "stats": normalizer.stats_dict,
        }
    )
    wandb.config.update(config.to_dict())

    # Create model save dir
    model_save_dir = Path(config.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    run_training_loop(
        config,
        env,
        device,
        actor,
        optimizer,
        trainloader,
        testloader,
        n_batches,
        lr_scheduler,
        early_stopper,
        model_save_dir,
    )

    wandb.finish()


def run_training_loop(
    config,
    env,
    device,
    actor,
    optimizer,
    trainloader,
    testloader,
    n_batches,
    lr_scheduler,
    early_stopper,
    model_save_dir,
):
    test_loss_mean = 0.0
    bc_loss_running_mean = 0.0

    tglobal = tqdm(range(config.num_epochs), desc="Epoch")
    best_success_rate = float("-inf")

    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        # batch loop
        tepoch = tqdm(trainloader, desc="Batch", leave=False, total=n_batches)
        for batch in tepoch:
            optimizer.zero_grad()

            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Get loss
            bc_loss, q_loss, value_loss = actor.compute_loss(batch)

            # Sum the losses
            loss = bc_loss + q_loss + value_loss

            # backward pass
            loss.backward()

            # Gradient clipping
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=config.clip_grad_norm
                )

            # optimizer step
            optimizer.step()
            lr_scheduler.step()

            # Update the target network
            actor.polyak_update_target(config.q_target_update_step)

            # logging
            loss = loss.item()
            bc_loss = bc_loss.item()
            q_loss = q_loss.item()
            value_loss = value_loss.item()

            epoch_loss.append(loss)
            bc_loss_running_mean = (
                0.9 * bc_loss_running_mean + 0.1 * bc_loss
            )  # EMA of the BC loss

            wandb.log(
                dict(
                    lr=lr_scheduler.get_last_lr()[0],
                    batch_loss=loss,
                    batch_bc_loss=bc_loss,
                    batch_q_loss=q_loss,
                    batch_value_loss=value_loss,
                    bc_loss_running_mean=bc_loss_running_mean,
                )
            )

            tepoch.set_postfix(loss=loss, bc_loss_running_mean=bc_loss_running_mean)

        tepoch.close()

        train_loss_mean = np.mean(epoch_loss)
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        # Evaluation loop
        test_losses = dict(bc_loss=list(), q_loss=list(), value_loss=list())
        test_tepoch = tqdm(testloader, desc="Test Batch", leave=False)
        for test_batch in test_tepoch:
            with torch.no_grad():
                # device transfer for test_batch
                test_batch = dict_apply(
                    test_batch, lambda x: x.to(device, non_blocking=True)
                )

                # Get test loss
                test_bc_loss, test_q_loss, test_value_loss = actor.compute_loss(
                    test_batch
                )

                test_losses["bc_loss"].append(test_bc_loss.item())
                test_losses["q_loss"].append(test_q_loss.item())
                test_losses["value_loss"].append(test_value_loss.item())

                # logging
                test_loss_cpu = (test_bc_loss + test_q_loss + test_value_loss).item()
                test_loss.append(test_loss_cpu)
                test_tepoch.set_postfix(loss=test_loss_cpu)

        test_tepoch.close()

        test_loss_mean = np.mean(test_loss)
        for key, value in test_losses.items():
            wandb.log({f"test_{key}": np.mean(value), "epoch": epoch_idx})

        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )

        wandb.log({"test_epoch_loss": test_loss_mean, "epoch": epoch_idx})

        # Early stopping
        if early_stopper.update(test_loss_mean):
            print(
                f"Early stopping at epoch {epoch_idx} as test loss did not improve for {early_stopper.patience} epochs."
            )
            break

        # Log the early stopping stats
        wandb.log(
            {
                "early_stopper/counter": early_stopper.counter,
                "early_stopper/best_loss": early_stopper.best_loss,
                "early_stopper/ema_loss": early_stopper.ema_loss,
            }
        )

        if (
            config.rollout.every != -1
            and (epoch_idx + 1) % config.rollout.every == 0
            and bc_loss_running_mean < config.rollout.loss_threshold
        ):
            # Checkpoint the model
            if config.checkpoint_model:
                save_path = str(model_save_dir / f"actor_{epoch_idx}.pt")
                torch.save(
                    actor.state_dict(),
                    save_path,
                )

            # Do no load the environment until we successfuly made it this far
            if env is None:
                env = get_env(
                    config.gpu_id,
                    obs_type=config.observation_type,
                    furniture=config.furniture,
                    num_envs=config.num_envs,
                    randomness=config.randomness,
                    resize_img=not config.augment_image,
                )
            best_success_rate = do_rollout_evaluation(
                config, env, model_save_dir, actor, best_success_rate, epoch_idx
            )
            tglobal.set_postfix(
                loss=train_loss_mean,
                test_loss=test_loss_mean,
                best_success_rate=best_success_rate,
            )

    tglobal.close()


def validate_config(config):
    assert (
        config.rollout.count % config.num_envs == 0
    ), "n_rollouts must be divisible by num_envs"

    assert (
        config.datasim_path.exists()
    ), f"Data path {config.datasim_path} does not exist"
    assert config.augment_image is False, "Image augmentation not implemented yet"
    assert not config.augment_image or config.observation_type == "image", (
        "Image augmentation only works with image observations. "
        f"Got observation type {config.observation_type}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--dryrun", "-d", action="store_true")
    parser.add_argument("--cpus", "-c", type=int, default=24)
    parser.add_argument("--wb-mode", "-w", type=str, default="online")
    args = parser.parse_args()

    def maybe(x, fb=1):
        return x if args.dryrun is False else fb

    data_base_dir = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))

    n_workers = min(args.cpus, os.cpu_count())
    num_envs = maybe(16, fb=2)

    config = ConfigDict()

    # TODO: Only for IDQL experiments
    config.freeze_policy = True

    config.action_horizon = 8
    config.actor_lr = 5e-7
    config.augment_image = False
    config.batch_size = args.batch_size
    config.beta_schedule = "squaredcos_cap_v2"
    config.checkpoint_model = False
    config.clip_grad_norm = 1
    config.clip_sample = True
    config.data_subset = maybe(None, fb=10)
    config.dataloader_workers = n_workers
    config.demo_source = "sim"
    config.down_dims = [256, 512, 1024]
    config.dryrun = args.dryrun
    config.furniture = "one_leg"
    config.gpu_id = args.gpu_id
    config.inference_steps = 16
    config.load_checkpoint_path = None
    config.mixed_precision = False
    config.num_diffusion_iters = 100
    config.num_envs = num_envs
    config.num_epochs = 200
    config.obs_horizon = 2
    config.observation_type = "feature"
    config.pred_horizon = 16
    config.prediction_type = "epsilon"
    config.randomness = "low"
    config.steps_per_epoch = maybe(200, fb=10)
    config.test_split = 0.1

    config.rollout = ConfigDict()
    config.rollout.every = maybe(10, fb=1)
    config.rollout.loss_threshold = maybe(10, fb=float("inf"))
    config.rollout.max_steps = maybe(750, fb=10)
    config.rollout.count = num_envs

    config.lr_scheduler = ConfigDict()
    config.lr_scheduler.name = "cosine"
    # config.lr_scheduler.warmup_pct = 0.025
    config.lr_scheduler.warmup_steps = 500

    config.vision_encoder = ConfigDict()
    config.vision_encoder.model = "vip"
    config.vision_encoder.freeze = True
    config.vision_encoder.normalize_features = False

    config.early_stopper = ConfigDict()
    config.early_stopper.smooth_factor = 0.9
    config.early_stopper.patience = 10

    # Regularization
    config.weight_decay = 1e-6
    config.feature_dropout = False
    config.noise_augment = False

    # Q-learning
    config.expectile = 0.75
    config.q_target_update_step = 0.005
    config.discount = 0.997
    config.critic_dropout = 0.5
    config.critic_lr = 5e-7
    config.critic_weight_decay = 5e-4
    config.critic_hidden_dims = [512, 256]
    config.n_action_samples = 10

    config.model_save_dir = "models"

    config.datasim_path = (
        data_base_dir / "processed/sim/feature_separate_small/vip/one_leg/data.zarr"
    )

    validate_config(config)

    print(f"Using data from {config.datasim_path}")

    main(config)
