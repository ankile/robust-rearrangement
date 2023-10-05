import os
from pathlib import Path
import furniture_bench
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from furniture_bench.envs.observation import DEFAULT_STATE_OBS
from src.data.dataset import FurnitureImageDataset, SimpleFurnitureDataset
from src.eval import calculate_success_rate
from src.gym import get_env
from src.models.unet import ConditionalUnet1D
from src.models.domain_adaptation import DomainClassifier
from tqdm import tqdm
import ipdb
from src.models.actor import ImageActor
import argparse

from src.models.vision import DinoEncoder
from ml_collections import ConfigDict
from accelerate import Accelerator


def main(config: ConfigDict):
    env = None

    # Init wandb
    wandb.init(
        project="furniture-diffusion",
        entity="ankile",
        config=config.to_dict(),
        mode="online" if not config.dryrun else "disabled",
    )
    config = wandb.config

    dataset = SimpleFurnitureDataset(
        dataset_path=config.datasim_path,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
    )

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

    accelerator = Accelerator(
        mixed_precision="fp16" if config.mixed_precision else "no",
    )

    device = accelerator.device

    enc = {
        "dinov2-base": DinoEncoder(size="base", freeze=True, device=device),
    }[config.vision_encoder]
    config.encoding_dim = enc.encoding_dim

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=config.action_dim,
        global_cond_dim=dataset.obs_dim * config.obs_horizon,
        down_dims=config.down_dims,
    ).to(device)

    actor = ImageActor(noise_pred_net, enc, config, stats)

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule=config.beta_schedule,
        # clip output to [-1,1] to improve stability
        clip_sample=config.clip_sample,
        # our network predicts noise (instead of denoised action)
        prediction_type=config.prediction_type,
    )

    # AdamW optimizer for noise_pred_net
    opt_noise = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
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

    # Prepare all the parts with accelerator
    noise_pred_net, enc, opt_noise, dataloader, lr_scheduler = accelerator.prepare(
        noise_pred_net, enc, opt_noise, dataloader, lr_scheduler
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
            nobs = batch["obs"].to(device)
            action = batch["action"].to(device)
            B = nobs.shape[0]

            # observation as FiLM conditioning
            # (B, obs_horizon, obs_dim)
            obs_cond = nobs[:, : config.obs_horizon, :]
            # (B, obs_horizon * obs_dim)
            obs_cond = obs_cond.flatten(start_dim=1)

            # observation as FiLM conditioning
            # (B, obs_horizon, obs_dim)
            # nobs = torch.cat((pos, feat), dim=2)
            # obs_cond = nobs[:, : config.obs_horizon, :]  # Not really doing anything
            # (B, obs_horizon * obs_dim)
            # obs_cond = obs_cond.flatten(start_dim=1)

            # sample noise to add to actions
            noise = torch.randn(action.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_action = noise_scheduler.add_noise(action, noise, timesteps)

            # Zero out the gradients
            opt_noise.zero_grad()

            # forward pass
            noise_pred = noise_pred_net(
                noisy_action, timesteps, global_cond=obs_cond.float()
            )
            loss = nn.functional.mse_loss(noise_pred, noise)

            # backward pass
            # loss.backward()
            accelerator.backward(loss)
            opt_noise.step()

            lr_scheduler.step()

            # Gradient clipping
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(noise_pred_net.parameters(), max_norm=1)

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

        if config.rollout_every != -1 and (epoch_idx + 1) % config.rollout_every == 0:
            if env is None:
                env = get_env(
                    config.gpu_id,
                    obs_type=config.observation_type,
                    furniture=config.furniture,
                    num_envs=1,
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
                torch.save(
                    noise_pred_net.state_dict(),
                    f"noise_pred_net.pth",
                )

                wandb.save("noise_pred_net.pth")

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--dryrun", "-d", action="store_true")
    args = parser.parse_args()

    data_base_dir = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))

    config = ConfigDict(
        dict(
            dryrun=args.dryrun,
            pred_horizon=16,
            obs_horizon=2,
            action_horizon=8,
            down_dims=[256, 512, 1024],
            batch_size=args.batch_size,
            num_epochs=100,
            num_diffusion_iters=100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
            actor_lr=1e-5,
            weight_decay=1e-6,
            lr_scheduler_type="cosine",
            lr_scheduler_warmup_steps=500,
            dataloader_workers=24,
            rollout_every=10 if args.dryrun is False else 1,
            n_rollouts=5 if args.dryrun is False else 1,
            n_envs=1,
            inference_steps=10,
            mixed_precision=True,
            clip_grad_norm=False,
            gpu_id=args.gpu_id,
            furniture="one_leg",
            observation_type="feature",
            rollout_max_steps=750 if args.dryrun is False else 10,
            demo_source="sim",
            vision_encoder="dinov2-base",
        )
    )
    config.datasim_path = (
        data_base_dir
        / "processed"
        / config.demo_source
        / config.observation_type
        / config.vision_encoder
        / "low"
        / config.furniture
        / "data.zarr"
    )

    print(f"Using data from {config.datasim_path}")

    main(config)
