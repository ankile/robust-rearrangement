import os
from pathlib import Path
import furniture_bench
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from diffusers.optimization import get_scheduler
from src.data.dataset import FurnitureImageDataset, FurnitureFeatureDataset, SimpleFurnitureDataset
from src.data.normalizer import StateActionNormalizer
from src.eval import calculate_success_rate
from src.gym import get_env
from tqdm import tqdm
from ipdb import set_trace as bp
from src.models.actor import DoubleImageActor
from src.data.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_apply
import argparse
from torch.utils.data import random_split


from ml_collections import ConfigDict


def main(config: ConfigDict):
    env = None
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Init wandb
    wandb.init(
        project="furniture-diffusion",
        entity="ankile",
        config=config.to_dict(),
        mode="online" if not config.dryrun else "disabled",
        notes="Run to see if downdims starting at 512 is better than 256",
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
            data_subset=config.data_subset,
        )

    else:
        raise ValueError(f"Unknown observation type: {config.observation_type}")

    # Split the dataset into train and test
    train_size = int(len(dataset) * config.test_split)
    test_size = len(dataset) - train_size
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

    if config.load_checkpoint_path is not None:
        print(f"Loading checkpoint from {config.load_checkpoint_path}")
        actor.load_state_dict(torch.load(config.load_checkpoint_path))

    # Update the config object with the observation dimension
    config.obs_dim = actor.obs_dim

    # save stats to wandb and update the config object
    wandb.log(
        {
            "num_samples": len(train_dataset),
            "num_episodes": int(len(dataset.episode_ends) * config.test_split),
            "stats": normalizer.stats_dict,
        }
    )
    wandb.config.update(config)

    # create dataloader
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

    testloader = FixedStepsDataloader(
        dataset=test_dataset,
        n_batches=10,
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

    lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.name)(
        optimizer=opt_noise,
        max_lr=config.actor_lr,
        epochs=config.num_epochs,
        steps_per_epoch=n_batches,
        pct_start=config.lr_scheduler.warmup,
        anneal_strategy="cos",
    )

    tglobal = tqdm(range(config.num_epochs), desc="Epoch")
    best_success_rate = float("-inf")

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
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=config.clip_grad_norm)

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
        tglobal.set_postfix(loss=train_loss_mean, test_loss=test_loss_mean)
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        # Evaluation loop
        test_tepoch = tqdm(testloader, desc="Test Batch", leave=False)
        for test_batch in test_tepoch:
            with torch.no_grad():
                # device transfer for test_batch
                test_batch = dict_apply(test_batch, lambda x: x.to(device, non_blocking=True))

                # Get test loss
                test_loss_val = actor.compute_loss(test_batch)

                # logging
                test_loss_cpu = test_loss_val.item()
                test_loss.append(test_loss_cpu)
                test_tepoch.set_postfix(loss=test_loss_cpu)

        test_loss_mean = np.mean(test_loss)
        wandb.log({"test_epoch_loss": test_loss_mean, "epoch": epoch_idx})
        test_tepoch.set_postfix(loss=train_loss_mean, test_loss=test_loss_mean)
        test_tepoch.close()

        if (
            config.rollout.every != -1
            and (epoch_idx + 1) % config.rollout.every == 0
            and np.mean(epoch_loss) < config.rollout.loss_threshold
        ):
            if env is None:
                env = get_env(
                    config.gpu_id,
                    obs_type=config.observation_type,
                    furniture=config.furniture,
                    num_envs=config.num_envs,
                    randomness=config.randomness,
                    resize_img=not config.augment_image,
                )

            # Perform a rollout with the current model
            success_rate = calculate_success_rate(
                env,
                actor,
                n_rollouts=config.rollout.count,
                epoch_idx=epoch_idx,
            )

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_path = str(model_save_dir / f"actor_best.pt")
                torch.save(
                    actor.state_dict(),
                    save_path,
                )

                wandb.save(save_path)
                wandb.log({"best_success_rate": best_success_rate})

            # Checkpoint the model
            save_path = str(model_save_dir / f"actor_{epoch_idx}.pt")
            torch.save(
                actor.state_dict(),
                save_path,
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
    num_envs = 1

    config = ConfigDict()

    config.action_horizon = 8
    config.actor_lr = 1e-4
    config.batch_size = args.batch_size
    config.beta_schedule = "squaredcos_cap_v2"
    config.clip_grad_norm = 1
    config.clip_sample = True
    config.data_subset = None if args.dryrun is False else 10
    config.dataloader_workers = n_workers
    config.demo_source = "sim"
    config.down_dims = [512, 1024, 2048]
    config.dryrun = args.dryrun
    config.furniture = "one_leg"
    config.gpu_id = args.gpu_id
    config.inference_steps = 16
    config.load_checkpoint_path = None
    config.mixed_precision = False
    config.num_diffusion_iters = 100
    config.num_envs = num_envs
    config.num_epochs = 200
    config.steps_per_epoch = 200 if args.dryrun is False else 10
    config.obs_horizon = 2
    config.observation_type = "feature"
    config.augment_image = False
    config.pred_horizon = 16
    config.prediction_type = "epsilon"
    config.randomness = "low"
    config.weight_decay = 1e-6
    config.test_split = 0.1

    config.rollout = ConfigDict()
    config.rollout.every = 10 if args.dryrun is False else 1
    config.rollout.loss_threshold = 0.01
    config.rollout.max_steps = 750 if args.dryrun is False else 10
    config.rollout.count = 10 if args.dryrun is False else num_envs

    config.lr_scheduler = ConfigDict()
    config.lr_scheduler.name = "OneCycleLR"
    config.lr_scheduler.warmup = 0.025

    config.vision_encoder = ConfigDict()
    config.vision_encoder.model = "r3m_18"
    config.vision_encoder.freeze = True

    config.model_save_dir = "models"

    assert config.rollout.count % config.num_envs == 0, "n_rollouts must be divisible by num_envs"

    config.datasim_path = (
        data_base_dir / f"processed/sim/feature_separate/{config.vision_encoder.model}/one_leg/data.zarr"
    )

    print(f"Using data from {config.datasim_path}")

    main(config)
