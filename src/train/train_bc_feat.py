import furniture_bench
import ml_collections
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from furniture_bench.envs.observation import DEFAULT_STATE_OBS
from src.data.dataset import SimpleFurnitureDataset, normalize_data, unnormalize_data
from src.eval import calculate_success_rate
from src.gym import get_env
from src.models.unet import ConditionalUnet1D
from tqdm import tqdm


def main(config: dict):
    # Init wandb
    wandb.init(project="furniture-diffusion", entity="ankile", config=config)
    config = wandb.config

    device = torch.device(f"cuda:{config.gpu_id}")

    # create env
    env = get_env(
        config.gpu_id, obs_type=config.observation_type, furniture=config.furniture
    )

    dataset = SimpleFurnitureDataset(
        dataset_path=config.dataset_path,
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
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=dataset.action_dim,
        global_cond_dim=dataset.obs_dim * config.obs_horizon,
        down_dims=config.down_dims,
    ).to(device)

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

    wandb.watch(noise_pred_net)

    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler_warmup_steps,
        num_training_steps=len(dataloader) * config.num_epochs,
    )

    tglobal = tqdm(range(config.num_epochs), desc="Epoch")
    best_success_rate = 0.0

    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch["obs"].to(device)
                naction = nbatch["action"].to(device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:, : config.obs_horizon, :]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                # forward pass
                optimizer.zero_grad()
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond
                )
                loss = nn.functional.mse_loss(noise_pred, noise)

                # backward pass
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                # Gradient clipping
                if config.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        noise_pred_net.parameters(), max_norm=1
                    )

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                wandb.log({"lr": lr_scheduler.get_last_lr()[0]})
                wandb.log({"batch_loss": loss_cpu})

                tepoch.set_postfix(loss=loss_cpu)

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        if (epoch_idx + 1) % config.rollout_every == 0:
            # Swap the EMA weights with the current model weights
            # ema.swap(noise_pred_net.parameters())

            # Perform a rollout with the current model
            success_rate = calculate_success_rate(
                env,
                noise_pred_net,
                stats,
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
    config = dict(
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=6,
        down_dims=[128, 512, 1024],
        batch_size=512,
        num_epochs=500,
        num_diffusion_iters=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
        lr=5e-4,
        weight_decay=1e-6,
        ema_power=0.75,
        lr_scheduler_type="cosine",
        lr_scheduler_warmup_steps=500,
        dataloader_workers=16,
        rollout_every=25,
        n_rollouts=5,
        inference_steps=10,
        ema_model=False,
        dataset_path="/mnt/batch/tasks/shared/LS_root/mounts/clusters/a100/data/processed/sim/feature/one_leg/data.zarr",
        mixed_precision=False,
        clip_grad_norm=False,
        gpu_id=1,
        furniture="one_leg",
        observation_type="feature",
        rollout_max_steps=750,
        demo_source="sim",
    )

    main(config)
