from re import A
import furniture_bench  # noqa: F401
from ml_collections import ConfigDict
from sklearn import base
import torch

import collections
import imageio
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp  # noqa: F401
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
import pickle

from src.behavior.base import Actor


import wandb


def rollout(
    env: FurnitureSimEnv,
    actor: Actor,
    rollout_max_steps: int,
    pbar: tqdm = None,
):
    # get first observation
    obs = env.reset()

    obs_horizon = actor.obs_horizon
    action_horizon = actor.action_horizon

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
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")

    # Define a noop tensor to use when done
    # It is zero everywhere except for the first element of the rotation
    # as the quaternion noop is (1, 0, 0, 0)
    # The elements are: (x, y, z) + (a, b, c, d) + (w,)
    noop = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32, device="cuda")

    step_idx = 0
    while not done.all():
        # Get the next actions from the actor
        action_pred = actor.action(obs_deque)
        curr_action = action_pred.clone()
        curr_action[done.nonzero()] = noop

        obs, reward, done, _ = env.step(curr_action)

        # save observations
        obs_deque.append(obs)

        # and reward/vis
        robot_states.append(obs["robot_state"].cpu())
        imgs1.append(obs["color_image1"].cpu())
        imgs2.append(obs["color_image2"].cpu())
        actions.append(curr_action.cpu())
        rewards.append(reward.cpu())

        # update progress bar
        step_idx += 1
        if pbar is not None:
            pbar.set_postfix(step=step_idx)

        if step_idx >= rollout_max_steps:
            done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")

        if done.all():
            break

    return (
        torch.stack(robot_states).transpose(0, 1),
        torch.stack(imgs1).transpose(0, 1),
        torch.stack(imgs2).transpose(0, 1),
        torch.stack(actions, dim=1),
        torch.cat(rewards, dim=1),
    )


def create_in_memory_mp4(np_images, fps=10):
    output = BytesIO()

    writer_options = {"fps": fps}
    writer_options["format"] = "mp4"
    writer_options["codec"] = "libx264"
    writer_options["pixelformat"] = "yuv420p"

    with imageio.get_writer(output, **writer_options) as writer:
        for img in np_images:
            writer.append_data(img)

    output.seek(0)
    return output


def save_raw_rollout(
    robot_states, imgs1, imgs2, actions, rewards, success, furniture, output_path
):
    observations = list()

    for robot_state, image1, image2, action, reward in zip(
        robot_states, imgs1, imgs2, actions, rewards
    ):
        observations.append(
            {
                "robot_state": robot_state,
                "color_image1": image1,
                "color_image2": image2,
            }
        )

    data = {
        "observations": observations,
        "actions": action.tolist(),
        "rewards": reward.tolist(),
        "success": success,
        "furniture": furniture,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


@torch.no_grad()
def calculate_success_rate(
    env: FurnitureSimEnv,
    actor: Actor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    gamma: float = 0.99,
    rollout_save_dir: str = None,
):
    tbl = wandb.Table(columns=["rollout", "success", "epoch", "return", "steps"])
    pbar = trange(
        n_rollouts,
        desc="Performing rollouts",
        leave=False,
    )
    # To save the rollouts as training data for later, we need to save:
    # observations
    #  - robot_state
    #  - color_image1
    #  - color_image2
    # actions
    # rewards

    n_success = 0

    all_robot_states = list()
    all_imgs1 = list()
    all_imgs2 = list()
    all_actions = list()
    all_rewards = list()

    for _ in range(n_rollouts // env.num_envs):
        # Perform a rollout with the current model
        robot_states, imgs1, imgs2, actions, rewards = rollout(
            env,
            actor,
            rollout_max_steps,
            pbar=pbar,
        )

        # Calculate the success rate
        success = rewards.sum(dim=1) > 0
        n_success += success.sum().item()

        # Save the results from the rollout
        all_robot_states.extend(robot_states)
        all_imgs1.extend(imgs1)
        all_imgs2.extend(imgs2)
        all_actions.extend(actions)
        all_rewards.extend(rewards)

        # Update progress bar
        pbar.update(env.num_envs)
        pbar.set_postfix(success=n_success)

    total_return = 0
    table_rows = []
    for rollout_idx in range(n_rollouts):
        # Get the rewards and images for this rollout
        robot_states = all_robot_states[rollout_idx].numpy()
        video1 = all_imgs1[rollout_idx].numpy()
        video2 = all_imgs2[rollout_idx].numpy()
        actions = all_actions[rollout_idx].numpy()
        rewards = all_rewards[rollout_idx].numpy()
        success = (rewards.sum() > 0).item()
        furniture = env.furniture_name

        # Stack the two videos side by side into a single video
        # and keep axes as (T, H, W, C)
        video = np.concatenate([video1, video2], axis=2)
        video = create_in_memory_mp4(video, fps=10)

        n_steps = np.argmax(rewards) + 1

        # Calculate the return for this rollout
        episode_return = np.sum(rewards * gamma ** np.arange(len(rewards)))
        total_return += episode_return

        table_rows.append(
            [
                wandb.Video(video, fps=10, format="mp4"),
                success,
                epoch_idx,
                episode_return,
                n_steps,
            ]
        )

        if rollout_save_dir is not None:
            output_path = rollout_save_dir / f"rollout_{rollout_idx}.pkl"

            # Save the raw rollout data
            save_raw_rollout(
                robot_states,
                video1,
                video2,
                actions,
                rewards,
                success,
                furniture,
                output_path,
            )

    # Sort the table rows by return (highest at the top)
    table_rows = sorted(table_rows, key=lambda x: x[3], reverse=True)

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
    model_save_dir: str,
    actor: Actor,
    best_success_rate: float,
    epoch_idx: int,
) -> float:
    rollout_save_dir = (
        Path(config.data_base_dir)
        / "raw"
        / "sim_rollouts"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    rollout_save_dir.mkdir(parents=True, exist_ok=True)

    success_rate = calculate_success_rate(
        env,
        actor,
        n_rollouts=config.rollout.count,
        rollout_max_steps=config.rollout.max_steps,
        epoch_idx=epoch_idx,
        gamma=config.discount,
        rollout_save_dir=rollout_save_dir,
    )

    if success_rate > best_success_rate:
        best_success_rate = success_rate
        save_path = str(model_save_dir / "actor_best.pt")
        torch.save(
            actor.state_dict(),
            save_path,
        )

        wandb.save(save_path)

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "best_success_rate": best_success_rate,
            "epoch": epoch_idx,
        }
    )

    return best_success_rate
