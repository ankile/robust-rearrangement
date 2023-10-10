import os
from pathlib import Path
import furniture_bench
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.data.dataset import FurnitureImageDataset, SimpleFurnitureDataset
from src.eval import calculate_success_rate
from src.gym import get_env
from tqdm import tqdm
from ipdb import set_trace as bp
from src.models.actor import DoubleImageActor
from src.common.pytorch_util import dict_apply
import argparse

from ml_collections import ConfigDict

model_save_dir = Path("models")


def main(config: ConfigDict):
    env = None
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Init wandb
    wandb.init(
        project="furniture-diffusion",
        entity="ankile",
        config=config.to_dict(),
        mode="online" if not config.dryrun else "disabled",
    )
    config = wandb.config

    if config.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_path=config.datasim_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
        )
    elif config.observation_type == "feature":
        dataset = SimpleFurnitureDataset(
            dataset_path=config.datasim_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
        )
    else:
        raise ValueError(f"Unknown observation type: {config.observation_type}")

    # Update the config object with the action and observation dimensions
    config.action_dim = dataset.action_dim
    config.obs_dim = dataset.obs_dim

    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # save stats to wandb
    wandb.log(
        {
            "num_samples": len(dataset),
            "num_episodes": len(dataset.episode_ends),
            "stats": stats,
        }
    )

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    actor = DoubleImageActor(
        device=device,
        resnet_size="18",
        config=config,
        stats=stats,
    )

    # AdamW optimizer for noise_net
    opt_noise = torch.optim.AdamW(
        params=actor.parameters(),
        lr=config.actor_lr,
        weight_decay=config.weight_decay,
    )

    n_batches = len(dataloader)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=opt_noise,
        num_warmup_steps=config.lr_scheduler_warmup_steps,
        num_training_steps=config.num_epochs * n_batches,
    )

    tglobal = tqdm(range(config.num_epochs), desc="Epoch")
    best_success_rate = 0.0

    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()

        # batch loop
        tepoch = tqdm(dataloader, desc="Batch", leave=False, total=n_batches)
        for batch in tepoch:
            # data normalized in dataset
            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Get loss
            loss = actor.compute_loss(batch)

            # backward pass
            # loss.backward()
            loss.backward()
            opt_noise.step()

            lr_scheduler.step()

            # Gradient clipping
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1)

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
            if config.dryrun:
                break

        tepoch.close()

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        if (
            config.rollout_every != -1
            and (epoch_idx + 1) % config.rollout_every == 0
            and np.mean(epoch_loss) < config.rollout_loss_threshold
        ):
            if env is None:
                env = get_env(
                    config.gpu_id,
                    obs_type=config.observation_type,
                    furniture=config.furniture,
                    num_envs=config.num_envs,
                )

            # Perform a rollout with the current model
            success_rate = calculate_success_rate(
                env,
                actor,
                config,
                epoch_idx,
            )

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                save_path = (
                    model_save_dir / f"actor_{config.furniture}_{wandb.run.name}.pt"
                )
                torch.save(
                    actor.state_dict(),
                    save_path,
                )

                wandb.save(save_path)

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--dryrun", "-d", action="store_true")
    args = parser.parse_args()

    data_base_dir = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))
    maybe = lambda x, fb=1: x if args.dryrun is False else fb

    config = ConfigDict(
        dict(
            action_horizon=8,
            actor_lr=1e-5,
            batch_size=args.batch_size,
            beta_schedule="squaredcos_cap_v2",
            clip_grad_norm=False,
            clip_sample=True,
            dataloader_workers=24,
            demo_source="sim",
            down_dims=[256, 512, 1024],
            dryrun=args.dryrun,
            furniture="one_leg",
            gpu_id=args.gpu_id,
            inference_steps=16,
            lr_scheduler_type="cosine",
            lr_scheduler_warmup_steps=500,
            mixed_precision=False,
            n_rollouts=10 if args.dryrun is False else 1,
            num_diffusion_iters=100,
            # Use 1 env for now
            num_envs=1,
            num_epochs=1_000,
            obs_horizon=2,
            observation_type="feature",
            pred_horizon=16,
            prediction_type="epsilon",
            rollout_every=50 if args.dryrun is False else 1,
            rollout_loss_threshold=maybe(0.01, 1e9),
            rollout_max_steps=750 if args.dryrun is False else 10,
            vision_encoder="resnet18",
            vision_encoder_pretrained=False,
            weight_decay=1e-6,
        )
    )

    assert (
        config.n_rollouts % config.num_envs == 0
    ), "n_rollouts must be divisible by num_envs"

    config.datasim_path = (
        data_base_dir
        / "processed"
        / config.demo_source
        / config.observation_type
        / config.vision_encoder
        / "data.zarr"
    )

    print(f"Using data from {config.datasim_path}")

    main(config)
