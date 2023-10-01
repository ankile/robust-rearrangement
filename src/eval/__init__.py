import torch

import collections

import numpy as np
from tqdm import tqdm, trange


import wandb


def rollout(
    env,
    actor,
    config,
    pbar=True,
):
    def unpack_reward(reward):
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu()

        return reward.item()

    # get first observation
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * config.obs_horizon,
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
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)

            # Get the next actions from the actor
            with torch.no_grad():
                action_pred = actor.action(obs_seq)

            # only take action_horizon number of actions
            start = config.obs_horizon - 1
            end = start + config.action_horizon
            action = action_pred[start:end, :]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _ = env.step(action[i])
                # save observations
                obs_deque.append(obs)
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
    actor,
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
            actor,
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
