# %%
import matplotlib.animation as animation
from datetime import datetime

import furniture_bench
from furniture_bench.envs.observation import DEFAULT_STATE_OBS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import random
import math

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import wandb

import zarr
from glob import glob
import pickle

cuda_id = 1

device = torch.device(f"cuda:{cuda_id}")

# %%
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

# 2. must reset before use
obs = env.reset()

# 3.
action = env.action_space.sample()

# 4. Standard gym step method
obs, reward, done, info = env.step(action)

robot_state_dim = obs["robot_state"].shape[1]
parts_poses_state_dim = obs["parts_poses"].shape[1]
obs_dim = robot_state_dim + parts_poses_state_dim
action_dim = action.shape[1]


# %%
def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


# dataset
class OneLegStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        # read from zarr dataset
        dataset = zarr.open(dataset_path, "r")
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            "action": dataset["actions"][:],
            # (N, obs_dim)
            "obs": dataset["observations"][:],
        }
        # Marks one-past the last index for each episode
        self.episode_ends = dataset["episode_ends"][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        return nsample


# %% [markdown]
# ## Define the model

# %%
from typing import Union


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


# %% [markdown]
# ## Train the Damn Thing

# %%
# use a seed >200 to avoid initial states seen in the training dataset
import collections


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
                    (B, config.pred_horizon, action_dim), device=device
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


# %%
def render_mp4(ims1, ims2, filename=None):
    # Initialize plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Function to update plot
    def update(num):
        ax1.clear()
        ax2.clear()
        ax1.axis("off")
        ax2.axis("off")

        img_array1 = ims1[num]
        if isinstance(img_array1, torch.Tensor):
            img_array1 = img_array1.squeeze(0).cpu().numpy()

        img_array2 = ims2[num]
        if isinstance(img_array2, torch.Tensor):
            img_array2 = img_array2.squeeze(0).cpu().numpy()

        ax1.imshow(img_array1)
        ax2.imshow(img_array2)

    frame_indices = range(0, len(ims1), 1)

    ani = animation.FuncAnimation(fig, update, frames=tqdm(frame_indices), interval=100)

    if not filename:
        filename = f"render-{datetime.now()}.mp4"

    ani.save(filename)


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


# %%
# config for wandb
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

dataset = OneLegStateDataset(
    dataset_path=config.dataset_path,
    pred_horizon=config.pred_horizon,
    obs_horizon=config.obs_horizon,
    action_horizon=config.action_horizon,
)

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
    input_dim=action_dim,
    global_cond_dim=obs_dim * config.obs_horizon,
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

# %% [markdown]
#
