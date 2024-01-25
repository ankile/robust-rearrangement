import furniture_bench  # noqa: F401
from ml_collections import ConfigDict
import torch

import collections
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp  # noqa: F401
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
import pickle

from typing import Union

from src.behavior.base import Actor
from src.visualization.render_mp4 import create_in_memory_mp4
from src.common.context import suppress_all_output
from src.common.tasks import furniture2idx
from src.common.files import trajectory_save_dir
from src.data_collection.io import save_raw_rollout
from src.data_processing.utils import resize, resize_crop

import wandb


def rollout(
    env: FurnitureSimEnv,
    actor: Actor,
    rollout_max_steps: int,
    pbar: tqdm = None,
):
    # get first observation
    with suppress_all_output(True):
        obs = env.reset()

    # Resize the images in the observation
    obs["color_image1"] = resize(obs["color_image1"])
    obs["color_image2"] = resize_crop(obs["color_image2"])

    obs_horizon = actor.obs_horizon

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon,
        maxlen=obs_horizon,
    )

    # save visualization and rewards
    robot_states = [obs["robot_state"].cpu()]
    imgs1 = [obs["color_image1"].cpu()]
    imgs2 = [obs["color_image2"].cpu()]
    actions = list()
    rewards = list()
    parts_poses = list()
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")

    # Define a noop tensor to use when done
    # It is zero everywhere except for the first element of the rotation
    # as the quaternion noop is (1, 0, 0, 0)
    # The elements are: (x, y, z) + (a, b, c, d) + (w,)
    noop = {
        "quat": torch.tensor(
            [0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32, device="cuda"
        ),
        "rot_6d": torch.tensor(
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=torch.float32, device="cuda"
        ),
    }[env.act_rot_repr]

    step_idx = 0
    while not done.all():
        # Get the next actions from the actor
        action_pred = actor.action(obs_deque)
        curr_action = action_pred.clone()
        curr_action[done.nonzero()] = noop

        obs, reward, done, _ = env.step(curr_action)

        # Resize the images in the observation
        obs["color_image1"] = resize(obs["color_image1"])
        obs["color_image2"] = resize_crop(obs["color_image2"])

        # Save observations for the policy
        obs_deque.append(obs)

        # Store the results for visualization and logging
        robot_states.append(obs["robot_state"].cpu())
        imgs1.append(obs["color_image1"].cpu())
        imgs2.append(obs["color_image2"].cpu())
        actions.append(curr_action.cpu())
        rewards.append(reward.cpu())
        parts_poses.append(obs["parts_poses"].cpu())

        # update progress bar
        step_idx += 1
        if pbar is not None:
            pbar.set_postfix(step=step_idx)
            pbar.update()

        if step_idx >= rollout_max_steps:
            done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")

        if done.all():
            break

    return (
        torch.stack(robot_states, dim=1),
        torch.stack(imgs1, dim=1),
        torch.stack(imgs2, dim=1),
        torch.stack(actions, dim=1),
        # Using cat here removes the singleton dimension
        torch.cat(rewards, dim=1),
        torch.stack(parts_poses, dim=1),
    )


@torch.no_grad()
def calculate_success_rate(
    env: FurnitureSimEnv,
    actor: Actor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    gamma: float = 0.99,
    rollout_save_dir: Union[str, None] = None,
    save_failures: bool = False,
    n_parts_assemble: Union[int, None] = None,
):
    def pbar_desc(self: tqdm, i: int, n_success: int):
        rnd = i + 1
        total = rnd * env.num_envs
        success_rate = n_success / total if total > 0 else 0
        self.set_description(
            f"Performing rollouts: round {rnd}/{n_rollouts//env.num_envs}, success: {n_success}/{total} ({success_rate:.1%})"
        )

    if n_parts_assemble is None:
        n_parts_assemble = len(env.furniture.should_be_assembled)

    tbl = wandb.Table(
        columns=["rollout", "success", "epoch", "reward", "return", "steps"]
    )
    pbar = trange(
        n_rollouts,
        desc="Performing rollouts",
        leave=False,
        total=rollout_max_steps * (n_rollouts // env.num_envs),
    )

    tqdm.pbar_desc = pbar_desc

    n_success = 0

    all_robot_states = list()
    all_imgs1 = list()
    all_imgs2 = list()
    all_actions = list()
    all_rewards = list()
    all_parts_poses = list()
    all_success = list()

    pbar.pbar_desc(0, n_success)
    for i in range(n_rollouts // env.num_envs):
        # Perform a rollout with the current model
        robot_states, imgs1, imgs2, actions, rewards, parts_poses = rollout(
            env,
            actor,
            rollout_max_steps,
            pbar=pbar,
        )

        # Calculate the success rate
        success = rewards.sum(dim=1) == n_parts_assemble
        n_success += success.sum().item()

        # Save the results from the rollout
        all_robot_states.extend(robot_states)
        all_imgs1.extend(imgs1)
        all_imgs2.extend(imgs2)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_parts_poses.extend(parts_poses)
        all_success.extend(success)

        # Update the progress bar
        pbar.pbar_desc(i, n_success)

    total_return = 0
    table_rows = []
    for rollout_idx in trange(n_rollouts, desc="Saving rollouts", leave=False):
        # Get the rewards and images for this rollout
        robot_states = all_robot_states[rollout_idx].numpy()
        video1 = all_imgs1[rollout_idx].numpy()
        video2 = all_imgs2[rollout_idx].numpy()
        actions = all_actions[rollout_idx].numpy()
        rewards = all_rewards[rollout_idx].numpy()
        parts_poses = all_parts_poses[rollout_idx].numpy()
        success = all_success[rollout_idx].item()
        furniture = env.furniture_name

        # Number of steps until success, i.e., the index of the final reward received
        n_steps = np.where(rewards == 1)[0][-1] + 1 if success else rollout_max_steps

        # Stack the two videos side by side into a single video
        # and keep axes as (T, H, W, C) (and cut off after rollout reaches success)
        video = np.concatenate([video1, video2], axis=2)[:n_steps]
        video = create_in_memory_mp4(video, fps=20)

        # Calculate the return for this rollout
        episode_return = np.sum(rewards * gamma ** np.arange(len(rewards)))
        total_return += episode_return

        table_rows.append(
            [
                wandb.Video(video, fps=20, format="mp4"),
                success,
                epoch_idx,
                np.sum(rewards),
                episode_return,
                n_steps,
            ]
        )

        if rollout_save_dir is not None and (save_failures or success):
            # Save the raw rollout data
            save_raw_rollout(
                robot_states[:n_steps],
                video1[:n_steps],
                video2[:n_steps],
                actions[:n_steps],
                rewards[:n_steps],
                parts_poses[:n_steps],
                success,
                furniture,
                rollout_save_dir,
            )

    # Sort the table rows by return (highest at the top)
    table_rows = sorted(table_rows, key=lambda x: x[4], reverse=True)

    for row in table_rows:
        tbl.add_data(*row)

    # Log the videos to wandb table
    wandb.log(
        {
            "rollouts": tbl,
            "epoch": epoch_idx,
            "epoch_mean_return": total_return / n_rollouts,
        }
    )

    pbar.close()

    return n_success / n_rollouts


def do_rollout_evaluation(
    config: ConfigDict,
    env: FurnitureSimEnv,
    save_rollouts: bool,
    actor: Actor,
    best_success_rate: float,
    epoch_idx: int,
) -> float:
    rollout_save_dir = None

    if save_rollouts:
        rollout_save_dir = trajectory_save_dir(
            environment="sim",
            task=config.furniture,
            demo_source="rollout",
            randomness=config.randomness,
            # Don't create here because we have to do it when we save anyway
            create=False,
        )

    actor.set_task(furniture2idx[config.furniture])

    success_rate = calculate_success_rate(
        env,
        actor,
        n_rollouts=config.rollout.count,
        rollout_max_steps=config.rollout.max_steps,
        epoch_idx=epoch_idx,
        gamma=config.discount,
        rollout_save_dir=rollout_save_dir,
        save_failures=config.rollout.save_failures,
    )

    best_success_rate = max(best_success_rate, success_rate)

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "best_success_rate": best_success_rate,
            "epoch": epoch_idx,
        }
    )

    return best_success_rate
