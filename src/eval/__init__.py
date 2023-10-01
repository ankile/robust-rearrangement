import torch

import collections

import numpy as np
from tqdm import tqdm, trange

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from src.data.dataset import normalize_data, unnormalize_data


import wandb


def rollout(
    env,
    noise_pred_net,
    stats,
    config,
    pbar=True,
):
    device = next(noise_pred_net.parameters()).device

    def get_obs(obs, obs_type):
        if obs_type == "state":
            return torch.cat([obs["robot_state"], obs["parts_poses"]], dim=-1).cpu()
        elif obs_type == "feature":
            return np.concatenate(
                [obs["robot_state"], obs["image1"], obs["image2"]], axis=-1
            )
        elif obs_type == "image":
            return np.concatenate([obs["image1"], obs["image2"]], axis=-1)
        else:
            raise NotImplementedError

    def unpack_reward(reward):
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu()

        return reward.item()

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
    obs_type = config.observation_type
    obs_deque = collections.deque(
        [get_obs(obs, obs_type)] * config.obs_horizon,
        maxlen=config.obs_horizon,
    )

    # save visualization and rewards
    imgs1 = [obs["color_image1"]]
    imgs2 = [obs["color_image2"]]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(
        total=config.rollout_max_steps, desc="Eval OneLeg State Env", disable=not pbar
    ) as pbar:
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
                noise_scheduler.set_timesteps(config.inference_steps)

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
                obs_deque.append(get_obs(obs, obs_type))
                # and reward/vis
                rewards.append(unpack_reward(reward))
                imgs1.append(obs["color_image1"])
                imgs2.append(obs["color_image2"])

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > config.rollout_max_steps:
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
    def stack_frames(frames):
        if isinstance(frames[0], torch.Tensor):
            return (
                torch.stack(frames, dim=0)
                .squeeze(1)
                .cpu()
                .numpy()
                .transpose((0, 3, 1, 2))
            )
        return np.stack(frames, axis=0)

    n_success = 0

    tbl = wandb.Table(columns=["rollout", "success"])
    pbar = trange(
        config.n_rollouts, desc="Performing rollouts", postfix=dict(success=0)
    )
    for rollout_idx in pbar:
        # Perform a rollout with the current model
        rewards, imgs1, imgs2 = rollout(
            env,
            noise_pred_net,
            stats,
            config,
            pbar=False,
        )

        # Calculate the success rate
        success = np.sum(rewards) > 0
        n_success += int(success)
        pbar.set_postfix(success=n_success)

        # Stack the images into a single tensor
        video1 = stack_frames(imgs1)
        video2 = stack_frames(imgs2)

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
