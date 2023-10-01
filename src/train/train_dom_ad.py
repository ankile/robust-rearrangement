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
from src.data.dataset import FurnitureImageDataset
from src.eval import calculate_success_rate
from src.gym import get_env
from src.models.unet import ConditionalUnet1D
from src.models.domain_adaptation import DomainClassifier
from tqdm import tqdm
import ipdb
from src.models.actor import ImageActor
import argparse


from vip import load_vip


def main(config: dict):
    # Init wandb
    wandb.init(
        project="furniture-diffusion",
        entity="ankile",
        config=config,
        # mode="disabled",
    )
    config = wandb.config

    device = torch.device(f"cuda:{config.gpu_id}")

    # create env
    env = get_env(
        config.gpu_id,
        obs_type=config.observation_type,
        furniture=config.furniture,
    )

    data_real = FurnitureImageDataset(
        dataset_path=config.datareal_path,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
    )

    data_sim = FurnitureImageDataset(
        dataset_path=config.datasim_path,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
    )

    # Update the config object with the action and observation dimensions
    config.action_dim = data_real.action_dim
    config.agent_pos_dim = data_real.agent_pos_dim

    # save training data statistics (min, max) for each dim
    stats_real = data_real.stats
    stats_sim = data_sim.stats

    # save stats to wandb
    wandb.log(
        {
            "real.num_samples": len(data_real),
            "real.num_episodes": len(data_real.episode_ends),
            "real.stats": stats_real,
            "sim.num_samples": len(data_sim),
            "sim.num_episodes": len(data_sim.episode_ends),
            "sim.stats": stats_sim,
        }
    )

    half_batch = config.batch_size // 2
    half_workers = config.dataloader_workers // 2

    # create dataloader
    loader_real = torch.utils.data.DataLoader(
        data_real,
        batch_size=half_batch,
        num_workers=half_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        # persistent_workers=True,
    )

    loader_sim = torch.utils.data.DataLoader(
        data_sim,
        batch_size=half_batch,
        num_workers=half_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        # persistent_workers=True,
    )

    # load pretrained VIP encoder
    vip = load_vip(device_id=config.gpu_id).module

    # Get the VIP encoder output dimension
    vip_out_dim = vip.convnet.fc.out_features
    config.encoding_dim = vip_out_dim

    for param in vip.parameters():
        param.requires_grad = False

    if config.unfreeze_encoder_layers is not None:
        for name, param in vip.named_parameters():
            if any(
                [name.startswith(layer) for layer in config.unfreeze_encoder_layers]
            ):
                print(f"Unfreezing {name}")
                param.requires_grad = True

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=config.action_dim,
        global_cond_dim=(config.agent_pos_dim + 2 * config.encoding_dim)
        * config.obs_horizon,
        down_dims=config.down_dims,
    ).to(device)

    actor = ImageActor(noise_pred_net, vip, config, stats_sim)

    # create domain classifier
    domain_classifier = DomainClassifier(input_dim=vip_out_dim * 2).to(device)

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

    # AdamW optimizer for noise_pred_net
    opt_noise = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=config.actor_lr,
        weight_decay=config.weight_decay,
    )

    # AdamW optimizer for VIP encoder
    opt_encoder = torch.optim.AdamW(
        params=vip.parameters(),
        lr=config.encoder_lr,
        weight_decay=config.weight_decay,
    )

    # AdamW optimizer for domain classifier
    opt_domain = torch.optim.AdamW(
        params=domain_classifier.parameters(),
        lr=config.domain_lr,
        weight_decay=config.weight_decay,
    )

    n_batches = min(len(loader_real), len(loader_sim))

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
        batches = zip(loader_real, loader_sim)
        # batch loop
        tepoch = tqdm(batches, desc="Batch", leave=False, total=n_batches)
        for batch_real, batch_sim in tepoch:
            # data normalized in dataset
            # device transfer
            # Concat along batch dimension for real and sim data
            # Batch has keys "agent_pos", "action", "image1", "image2"
            pos_real = batch_real["agent_pos"].to(device)
            pos_sim = batch_sim["agent_pos"].to(device)
            pos = torch.cat((pos_real, pos_sim), dim=0)

            action_real = batch_sim["action"].to(device)
            action_sim = batch_sim["action"].to(device)
            action = torch.cat((action_real, action_sim), dim=0)

            img1_real = batch_real["image1"].to(device)
            img1_sim = batch_sim["image1"].to(device)
            img1 = torch.cat((img1_real, img1_sim), dim=0)

            feat1 = vip(img1.reshape(-1, 3, 224, 224)).reshape(
                -1, config.obs_horizon, vip_out_dim
            )

            img2_real = batch_real["image2"].to(device)
            img2_sim = batch_sim["image2"].to(device)
            img2 = torch.cat((img2_real, img2_sim), dim=0)
            feat2 = vip(img2.reshape(-1, 3, 224, 224)).reshape(
                -1, config.obs_horizon, vip_out_dim
            )

            # Concat features along the feature dimension to go from 2 * (B, obs_horizon, 1024) to (B, obs_horizon, 2048)
            feat = torch.cat((feat1, feat2), dim=2)

            # Takes in (B, obs_horizon, 2048) and outputs (B, obs_horizon, 1) squeeze to --> (B, obs_horizon)
            domain_pred = domain_classifier(feat).squeeze()

            # Create domain labels of size (B, obs_horizon)
            domain_y = (
                torch.cat(
                    [torch.ones(img1_real.shape[0]), torch.zeros(img1_sim.shape[0])]
                )
                .repeat(config.obs_horizon, 1)
                .T.to(device)
            )
            # ipdb.set_trace()

            # Todo: Add the image features
            B = pos.shape[0]

            # observation as FiLM conditioning
            # (B, obs_horizon, obs_dim)
            nobs = torch.cat((pos, feat), dim=2)
            obs_cond = nobs[:, : config.obs_horizon, :]  # Not really doing anything
            # (B, obs_horizon * obs_dim)
            obs_cond = obs_cond.flatten(start_dim=1)

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
            opt_encoder.zero_grad()
            opt_domain.zero_grad()

            # forward pass
            noise_pred = noise_pred_net(
                noisy_action, timesteps, global_cond=obs_cond.float()
            )
            diffusion_loss = nn.functional.mse_loss(noise_pred, noise)

            adv_loss = nn.functional.binary_cross_entropy_with_logits(
                domain_pred, domain_y
            )
            ratio = diffusion_loss.item() / (adv_loss.item() + 1e-8)
            dynamic_lambda = config.adv_lambda * ratio

            loss = diffusion_loss + dynamic_lambda * adv_loss

            adv_accuracy = (domain_pred > 0.5).eq(domain_y).sum().item() / (
                domain_pred.shape[0] * domain_pred.shape[1]
            )

            # backward pass
            loss.backward()
            opt_noise.step()

            # If the vision encoder produces easily classifiable features, train it
            if adv_accuracy > config.adversarial_accuracy_threshold:
                opt_encoder.step()

            # If domain classifier is not accurate enough, train it
            if adv_accuracy <= config.adversarial_accuracy_threshold:
                opt_domain.step()

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
                    adv_loss=adv_loss.item(),
                    diffusion_loss=diffusion_loss.item(),
                    adv_accuracy=adv_accuracy,
                )
            )

            tepoch.set_postfix(loss=loss_cpu)

        tepoch.close()

        tglobal.set_postfix(loss=np.mean(epoch_loss))
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        if config.rollout_every != -1 and (epoch_idx + 1) % config.rollout_every == 0:
            # Swap the EMA weights with the current model weights
            # ema.swap(noise_pred_net.parameters())

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

            # Swap the EMA weights back
            # ema.swap(noise_pred_net.parameters())

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=8)

    args = parser.parse_args()

    data_base_dir = Path(os.environ.get("FURNITURE_DATADIR", "/data"))
    print(f"Using data from {data_base_dir}")

    config = dict(
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=6,
        down_dims=[128, 512, 1024],
        batch_size=args.batch_size,
        num_epochs=200,
        num_diffusion_iters=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
        actor_lr=1e-5,
        encoder_lr=1e-6,
        unfreeze_encoder_layers=["layer4", "fc"],
        domain_lr=1e-5,
        weight_decay=1e-6,
        ema_power=0.75,
        lr_scheduler_type="cosine",
        lr_scheduler_warmup_steps=500,
        dataloader_workers=16,
        rollout_every=1,
        n_rollouts=5,
        inference_steps=10,
        ema_model=False,
        datareal_path=data_base_dir / "processed/real/image/low/one_leg/data.zarr",
        datasim_path=data_base_dir / "processed/sim/image/low/one_leg/data.zarr",
        mixed_precision=False,
        clip_grad_norm=False,
        gpu_id=args.gpu_id,
        furniture="one_leg",
        observation_type="image",
        rollout_max_steps=1_000,
        demo_source="mix",
        adv_lambda=1.0,
        adversarial_accuracy_threshold=0.75,
    )

    main(config)
