import furniture_bench
from furniture_bench.envs.observation import DEFAULT_STATE_OBS

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import collections

import gym
import numpy as np
from tqdm import tqdm, trange

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusers.optimization import get_scheduler

import wandb


from src.data.dataset import SimpleFurnitureDataset, normalize_data, unnormalize_data
from src.models.networks import ConditionalUnet1D

cuda_id = 1

device = torch.device(f"cuda:{cuda_id}")


env = gym.make(
    "FurnitureSim-v0",
    furniture="one_leg",  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
    num_envs=1,  # Number of parallel environments.
    resize_img=True,  # If true, images are resized to 224 x 224.
    concat_robot_state=True,  # If true, robot state is concatenated to the observation.
    obs_keys=DEFAULT_STATE_OBS
    + ["color_image1", "color_image2"],  # Specifies the observation keys.
    headless=True,  # If true, simulation runs without GUI.
    compute_device_id=cuda_id,
    graphics_device_id=cuda_id,
    init_assembled=False,  # If true, the environment is initialized with assembled furniture.
    np_step_out=False,  # If true, env.step() returns Numpy arrays.
    channel_first=False,  # If true, images are returned in channel first format.
    randomness="low",  # Level of randomness in the environment [low | med | high].
    high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
    save_camera_input=False,  # If true, the initial camera inputs are saved.
    record=False,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
    max_env_steps=3000,  # Maximum number of steps per episode.
    act_rot_repr="quat",  # Representation of rotation for action space. Options are 'quat' and 'axis'.
)


def rollout(
    env,
    noise_pred_net,
    stats,
    inference_steps,
    config,
    max_steps=500,
    pbar=True,
):
    # env.seed(10_000)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        prediction_type=config.prediction_type,
    )

    # get first observation
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [torch.cat([obs["robot_state"], obs["parts_poses"]], dim=-1).cpu()]
        * config.obs_horizon,
        maxlen=config.obs_horizon,
    )
    # save visualization and rewards
    imgs1 = [obs["color_image1"]]
    imgs2 = [obs["color_image2"]]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval OneLeg State Env", disable=not pbar) as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation

            nobs = normalize_data(obs_seq, stats=stats["obs"])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, config.pred_horizon, config.action_dim), device=device
                )
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(inference_steps)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = noise_pred_net(
                        sample=naction, timestep=k, global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to("cpu").numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats["action"])

            # only take action_horizon number of actions
            start = config.obs_horizon - 1
            end = start + config.action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = env.step(action[i])
                # save observations
                obs_deque.append(
                    torch.cat([obs["robot_state"], obs["parts_poses"]], dim=-1).cpu()
                )
                # and reward/vis
                rewards.append(reward.cpu().item())
                imgs1.append(obs["color_image1"])
                imgs2.append(obs["color_image2"])

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    return rewards, imgs1, imgs2


def calculate_success_rate(
    env,
    noise_pred_net,
    stats,
    config,
    epoch_idx,
):
    n_success = 0

    tbl = wandb.Table(columns=["rollout", "success"])
    for rollout_idx in trange(config.n_rollouts, desc="Performing rollouts"):
        # Perform a rollout with the current model
        rewards, imgs1, imgs2 = rollout(
            env,
            noise_pred_net,
            stats,
            config.inference_steps,
            config,
            pbar=False,
        )

        # Calculate the success rate
        success = np.sum(rewards) > 0
        n_success += int(success)

        # Stack the images into a single tensor
        video1 = (
            torch.stack(imgs1, dim=0).squeeze(1).cpu().numpy().transpose((0, 3, 1, 2))
        )
        video2 = (
            torch.stack(imgs2, dim=0).squeeze(1).cpu().numpy().transpose((0, 3, 1, 2))
        )

        # Stack the two videoes side by side into a single video
        video = np.concatenate((video1, video2), axis=3)

        tbl.add_data(wandb.Video(video, fps=10), success)

    # Log the videos to wandb table
    wandb.log(
        {
            "rollouts": tbl,
            "epoch": epoch_idx,
        }
    )

    # Log the success rate to wandb
    wandb.log({"success_rate": n_success / config.n_rollouts, "epoch": epoch_idx})

    return n_success / config.n_rollouts


config = dict(
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=6,
    down_dims=[256, 512, 1024],
    batch_size=4096,
    num_epochs=100,
    num_diffusion_iters=100,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=True,
    prediction_type="epsilon",
    lr=5e-4,
    weight_decay=1e-6,
    ema_power=0.75,
    lr_scheduler_type="cosine",
    lr_scheduler_warmup_steps=500,
    dataloader_workers=8,
    rollout_every=1,
    n_rollouts=10,
    inference_steps=10,
    ema_model=False,
    dataset_path="demos.zarr",
    mixed_precision=True,
    clip_grad_norm=False,
)

# Init wandb
wandb.init(project="furniture-diffusion", entity="ankile", config=config)
config = wandb.config

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

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
# ema = EMAModel(parameters=noise_pred_net.parameters(), power=config.ema_power)
scaler = GradScaler()

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

            optimizer.zero_grad()

            with autocast(enabled=config.mixed_precision):
                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond
                )

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(noise_pred_net.parameters(), max_norm=1)

            # Optimizer step and update scaler
            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            # ema.step(noise_pred_net.parameters())

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            wandb.log({"lr": lr_scheduler.get_last_lr()[0]})
            wandb.log({"batch_loss": loss_cpu})

            tepoch.set_postfix(loss=loss_cpu)

    tglobal.set_postfix(loss=np.mean(epoch_loss))
    wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

    if epoch_idx % config.rollout_every == 0:
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

        # Swap the EMA weights back
        # ema.swap(noise_pred_net.parameters())

tglobal.close()
wandb.finish()
