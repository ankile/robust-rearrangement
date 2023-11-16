import furniture_bench
import torch

import collections
import imageio
from io import BytesIO

import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

from src.models.actor import DoubleImageActor


import wandb


def rollout(
    env: FurnitureSimEnv,
    actor: DoubleImageActor,
    rollout_max_steps: int,
    pbar=True,
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
    imgs1 = [obs["color_image1"].cpu()]
    imgs2 = [obs["color_image2"].cpu()]
    rewards = list()
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")

    # Define a noop tensor to use when done
    # It is zero everywhere except for the first element of the rotation
    # as the quaternion noop is (1, 0, 0, 0)
    # The elements are: (x, y, z) + (a, b, c, d) + (w,)
    noop = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32, device="cuda")

    step_idx = 0

    pbar = tqdm(
        total=rollout_max_steps,
        desc="Eval OneLeg State Env",
        disable=not pbar,
    )
    while not done.all():
        # Get the next actions from the actor
        action_pred = actor.action(obs_deque)

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[:, start:end, :]
        # (num_envs, action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(action.shape[1]):
            # stepping env
            curr_action = action[:, i, :].clone()
            curr_action[done.nonzero()] = noop

            obs, reward, done, _ = env.step(curr_action)

            # save observations
            obs_deque.append(obs)

            # and reward/vis
            rewards.append(reward.cpu())
            imgs1.append(obs["color_image1"].cpu())
            imgs2.append(obs["color_image2"].cpu())

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx >= rollout_max_steps:
                done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")

            if done.all():
                break

    return (
        torch.cat(rewards, dim=1),
        torch.stack(imgs1).transpose(0, 1),
        torch.stack(imgs2).transpose(0, 1),
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


@torch.no_grad()
def calculate_success_rate(
    env: FurnitureSimEnv,
    actor: DoubleImageActor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    gamma: float = 0.99,
):
    tbl = wandb.Table(columns=["rollout", "success", "epoch", "return"])
    pbar = trange(
        n_rollouts,
        desc="Performing rollouts",
        postfix=dict(success=0),
        leave=False,
    )
    n_success = 0
    all_rewards = list()
    all_imgs1 = list()
    all_imgs2 = list()

    for _ in range(n_rollouts // env.num_envs):
        # Perform a rollout with the current model
        rewards, imgs1, imgs2 = rollout(
            env,
            actor,
            rollout_max_steps,
            pbar=False,
        )

        # Calculate the success rate
        success = rewards.sum(dim=1) > 0
        n_success += success.sum().item()

        # Save the results from the rollout
        all_rewards.extend(rewards)
        all_imgs1.extend(imgs1)
        all_imgs2.extend(imgs2)

        # Update progress bar
        pbar.update(env.num_envs)
        pbar.set_postfix(success=n_success)

    total_return = 0
    for rollout_idx in range(n_rollouts):
        # Get the rewards and images for this rollout
        rewards = all_rewards[rollout_idx].numpy()
        video1 = all_imgs1[rollout_idx].numpy()
        video2 = all_imgs2[rollout_idx].numpy()

        # Stack the two videoes side by side into a single video
        # and keep axes as (T, H, W, C)
        video = np.concatenate([video1, video2], axis=2)
        video = create_in_memory_mp4(video, fps=10)

        success = (rewards.sum() > 0).item()
        episode_return = np.sum(rewards[::-1] * gamma ** np.arange(len(rewards)))
        total_return += episode_return

        tbl.add_data(
            wandb.Video(video, fps=10, format="mp4"), success, epoch_idx, episode_return
        )

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
    config, env, model_save_dir, actor, best_success_rate, epoch_idx
) -> float:
    # Perform a rollout with the current model
    success_rate = calculate_success_rate(
        env,
        actor,
        n_rollouts=config.rollout.count,
        rollout_max_steps=config.rollout.max_steps,
        epoch_idx=epoch_idx,
        gamma=config.discount,
    )

    if success_rate > best_success_rate:
        best_success_rate = success_rate
        # save_path = str(model_save_dir / f"actor_best.pt")
        # torch.save(
        #     actor.state_dict(),
        #     save_path,
        # )

        # wandb.save(save_path)

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "best_success_rate": best_success_rate,
            "epoch": epoch_idx,
        }
    )

    return best_success_rate
