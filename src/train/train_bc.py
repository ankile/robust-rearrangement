import os
from pathlib import Path
import furniture_bench
import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from src.data.dataset import (
    FurnitureImageDataset,
    FurnitureFeatureDataset,
)
from src.data.normalizer import StateActionNormalizer
from src.eval import do_rollout_evaluation
from src.gym import get_env
from tqdm import tqdm
from ipdb import set_trace as bp
from src.models.actor import DoubleImageActor
from src.data.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_apply
import argparse
from torch.utils.data import random_split, DataLoader
from src.common.earlystop import EarlyStopper

from ml_collections import ConfigDict


def main(config: ConfigDict):
    env = None
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Init wandb
    wandb.init(
        project="robot-rearrangement",
        entity="robot-rearrangement",
        config=config.to_dict(),
        mode="online" if not config.dryrun else "disabled",
        notes="Fine-tune with unfreezed encoder and image augmentation (translation), take 2.",
    )

    # Create model save dir
    model_save_dir = Path(config.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    normalizer = StateActionNormalizer()

    if config.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_path=config.datasim_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            normalizer=normalizer,
            augment_image=config.augment_image,
            data_subset=config.data_subset,
        )
    elif config.observation_type == "feature":
        dataset = FurnitureFeatureDataset(
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
    actor = DoubleImageActor(
        device=device,
        encoder_name=config.vision_encoder.model,
        freeze_encoder=config.vision_encoder.freeze,
        normalizer=normalizer,
        config=config,
    )

    # Watch the model
    wandb.watch(actor, log="all")

    if config.load_checkpoint_path is not None:
        print(f"Loading checkpoint from {config.load_checkpoint_path}")
        actor.load_state_dict(torch.load(config.load_checkpoint_path))

    # Update the config object with the observation dimension
    config.obs_dim = actor.obs_dim

    # save stats to wandb and update the config object
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

    # AdamW optimizer for noise_net
    opt_noise = torch.optim.AdamW(
        params=actor.parameters(),
        lr=config.actor_lr,
        weight_decay=config.weight_decay,
    )

    n_batches = len(trainloader)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.name,
        optimizer=opt_noise,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * config.num_epochs,
    )

    tglobal = tqdm(range(config.num_epochs), desc="Epoch")
    best_success_rate = float("-inf")

    early_stopper = EarlyStopper(
        patience=config.early_stopper.patience,
        smooth_factor=config.early_stopper.smooth_factor,
    )

    # Train loop
    test_loss_mean = 0.0
    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        # batch loop
        tepoch = tqdm(trainloader, desc="Batch", leave=False, total=n_batches)
        for batch in tepoch:
            opt_noise.zero_grad()

            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Get loss
            loss = actor.compute_loss(batch)

            # backward pass
            loss.backward()

            # Gradient clipping
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=config.clip_grad_norm
                )

            # optimizer step
            opt_noise.step()
            lr_scheduler.step()

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            wandb.log(
                dict(
                    lr=lr_scheduler.get_last_lr()[0],
                    batch_loss=loss_cpu,
                )
            )

            tepoch.set_postfix(loss=loss_cpu)

        tepoch.close()

        train_loss_mean = np.mean(epoch_loss)
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        # Evaluation loop
        test_tepoch = tqdm(testloader, desc="Test Batch", leave=False)
        for test_batch in test_tepoch:
            with torch.no_grad():
                # device transfer for test_batch
                test_batch = dict_apply(
                    test_batch, lambda x: x.to(device, non_blocking=True)
                )

                # Get test loss
                test_loss_val = actor.compute_loss(test_batch)

                # logging
                test_loss_cpu = test_loss_val.item()
                test_loss.append(test_loss_cpu)
                test_tepoch.set_postfix(loss=test_loss_cpu)

        test_tepoch.close()

        test_loss_mean = np.mean(test_loss)
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
            and np.mean(epoch_loss) < config.rollout.loss_threshold
        ):
            # Checkpoint the model
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

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--dryrun", "-d", action="store_true")
    parser.add_argument("--cpus", "-c", type=int, default=24)
    parser.add_argument("--wb-mode", "-w", type=str, default="online")
    args = parser.parse_args()

    data_base_dir = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))
    maybe = lambda x, fb=1: x if args.dryrun is False else fb

    n_workers = min(args.cpus, os.cpu_count())
    num_envs = maybe(16, fb=2)

    config = ConfigDict()

    config.action_horizon = 8
    config.actor_lr = 5e-5
    config.augment_image = True
    config.batch_size = args.batch_size
    config.beta_schedule = "squaredcos_cap_v2"
    config.clip_grad_norm = False
    config.clip_sample = True
    config.data_subset = None if args.dryrun is False else 10
    config.dataloader_workers = n_workers
    config.demo_source = "sim"
    config.down_dims = [256, 512, 1024]
    config.dryrun = args.dryrun
    config.furniture = "one_leg"
    config.gpu_id = args.gpu_id
    config.inference_steps = 16
    config.load_checkpoint_path = (
        "/data/scratch/ankile/furniture-diffusion/models/glorious-bee-13/actor_199.pt"
    )
    config.mixed_precision = False
    config.num_diffusion_iters = 100
    config.num_envs = num_envs
    config.num_epochs = 500
    config.obs_horizon = 2
    config.observation_type = "image"
    config.pred_horizon = 16
    config.prediction_type = "epsilon"
    config.randomness = "low"
    config.steps_per_epoch = 200 if args.dryrun is False else 10
    config.test_split = 0.1

    config.rollout = ConfigDict()
    config.rollout.every = 10 if args.dryrun is False else 1
    config.rollout.loss_threshold = 0.03 if args.dryrun is False else float("inf")
    config.rollout.max_steps = 750 if args.dryrun is False else 10
    config.rollout.count = num_envs

    config.lr_scheduler = ConfigDict()
    config.lr_scheduler.name = "cosine"
    # config.lr_scheduler.warmup_pct = 0.025
    config.lr_scheduler.warmup_steps = 500

    config.vision_encoder = ConfigDict()
    config.vision_encoder.model = "r3m_18"
    config.vision_encoder.freeze = False
    config.vision_encoder.normalize_features = False

    config.early_stopper = ConfigDict()
    config.early_stopper.smooth_factor = 0.9
    config.early_stopper.patience = 10

    # Regularization
    config.weight_decay = 1e-6
    config.feature_dropout = False
    config.noise_augment = False

    config.model_save_dir = "models"

    assert (
        config.rollout.count % config.num_envs == 0
    ), "n_rollouts must be divisible by num_envs"

    config.datasim_path = "/data/scratch/ankile/furniture-data/data/processed/sim/image_small/one_leg/data.zarr"

    print(f"Using data from {config.datasim_path}")

    main(config)
