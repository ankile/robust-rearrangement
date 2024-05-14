from pathlib import Path
import furniture_bench  # noqa

from ipdb import set_trace as bp


from collections import deque
from typing import Literal, Tuple

import random
import time
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf
from src.behavior.base import Actor

from src.eval.load_model import load_bc_actor

from src.gym.env_rl_wrapper import ResidualPolicyEnvWrapper


from src.gym.furniture_sim_env import (
    FurnitureRLSimEnv,
)

from furniture_bench.envs.observation import DEFAULT_STATE_OBS
import numpy as np
from src.common.context import suppress_all_output
from src.models.residual import ResidualPolicy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torch.distributions.normal import Normal

import wandb


from src.gym import turn_off_april_tags

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    steps_per_iteration: int,
    gamma: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - dones[-1].to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


@hydra.main(config_path="../config/rl", config_name="residual_ppo", version_base="1.2")
def main(cfg: DictConfig):

    # TRY NOT TO MODIFY: seeding
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    run_name = f"{int(time.time())}__residual_ppo__{cfg.residual_policy._target_.split('.')[-1]}__{cfg.seed}"

    run_directory = f"runs/debug-residual_ppo-residual-8"
    run_directory += "-delete" if cfg.debug else ""
    print(f"Run directory: {run_directory}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda")
    action_type = cfg.action_type

    turn_off_april_tags()

    # env setup
    with suppress_all_output(False):
        kwargs = dict(
            act_rot_repr=cfg.act_rot_repr,
            action_type=action_type,
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture="one_leg",
            gpu_id=0,
            headless=cfg.headless,
            num_envs=cfg.num_envs,
            observation_space="state",
            randomness="low",
            max_env_steps=100_000_000,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        env: FurnitureRLSimEnv = FurnitureRLSimEnv(**kwargs)

    env.max_force_magnitude = 0.05
    env.max_torque_magnitude = 0.0025

    # Load the behavior cloning actor
    # TODO: The actor should keep tack of its own deque of observations
    # Similar to how it keeps track of a deque of actions
    # bc_actor: Actor = load_bc_actor("ankile/one_leg-mlp-state-1/runs/1ghcw9lu")
    bc_actor: Actor = load_bc_actor("ankile/one_leg-diffusion-state-1/runs/7623y5vn")

    env: ResidualPolicyEnvWrapper = ResidualPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        reset_on_failure=cfg.reset_on_failure,
    )
    env.set_normalizer(bc_actor.normalizer)

    # Residual policy setup
    residual_policy: ResidualPolicy = hydra.utils.instantiate(
        cfg.residual_policy,
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
    )

    residual_policy.to(device)

    optimizer = optim.AdamW(
        residual_policy.parameters(),
        lr=cfg.learning_rate,
        eps=1e-5,
        weight_decay=1e-6,
    )

    steps_per_iteration = cfg.data_collection_steps

    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(
        f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}"
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode="online" if not cfg.debug else "disabled",
    )

    # ALGO Logic: Storage setup
    best_eval_success_rate = 0.0
    running_mean_success_rate = 0.0

    obs = torch.zeros((steps_per_iteration, cfg.num_envs, residual_policy.obs_dim))
    actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape)
    logprobs = torch.zeros((steps_per_iteration, cfg.num_envs))
    rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    dones = torch.zeros((steps_per_iteration, cfg.num_envs))
    values = torch.zeros((steps_per_iteration, cfg.num_envs))

    global_step = 0
    iteration = 0
    start_time = time.time()
    # bp()
    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    base_observation_deque = deque(
        [next_obs for _ in range(bc_actor.obs_horizon)],
        maxlen=bc_actor.obs_horizon,
    )

    # First get the base normalized action
    base_naction = bc_actor.action_normalized(base_observation_deque)

    # Make the observation for the residual policy by concatenating the state and the base action
    next_obs = env.process_obs(next_obs)
    next_residual_obs = torch.cat([next_obs, base_naction], dim=-1)

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    while global_step < cfg.total_timesteps:
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")

        if iteration % cfg.eval_interval == 0:
            eval_mode = True
            # Also reset the env to have more consistent results
            next_obs = env.reset()
        else:
            eval_mode = False
        print(f"Eval mode: {eval_mode}")

        # Annealing the learning rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (global_step - 1) / cfg.total_timesteps
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, steps_per_iteration):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs

            dones[step] = next_done
            obs[step] = next_residual_obs

            with torch.no_grad():
                residual_naction_samp, logprob, _, value, action_mean = (
                    residual_policy.get_action_and_value(next_residual_obs)
                )

            residual_naction = residual_naction_samp if not eval_mode else action_mean
            naction = base_naction + residual_naction * cfg.residual_policy.action_scale
            next_obs, reward, next_done, truncated, infos = env.step(naction)

            # Add the observation to the deque
            base_observation_deque.append(next_obs)

            # Get the base normalized action
            base_naction = bc_actor.action_normalized(base_observation_deque)

            # Process the obs for the residual policy
            next_obs = env.process_obs(next_obs)
            next_residual_obs = torch.cat([next_obs, base_naction], dim=-1)

            values[step] = value.flatten().cpu()
            actions[step] = residual_naction.cpu()
            logprobs[step] = logprob.cpu()
            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()}"
                )

        # Calculate the discounted rewards
        # TODO: Change this so that it takes into account cyclic resets and multiple episodes
        # per experience collection iteration

        reward_mask = (rewards.sum(dim=0) > 0).float()
        success_rate = reward_mask.mean().item()

        running_mean_success_rate = 0.5 * running_mean_success_rate + 0.5 * success_rate

        print(f"SR: {success_rate:.4%}, SR mean: {running_mean_success_rate:.4%}")

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue
            wandb.log({"eval/success_rate": success_rate}, step=global_step)

            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"eval_best.pt")
                torch.save(
                    {
                        "model_state_dict": residual_policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "iteration": iteration,
                    },
                    model_path,
                )

                wandb.save(model_path)
                print(f"Evaluation success rate improved. Model saved to {model_path}")

            # Also reset the env before the next iteration
            next_obs = env.reset()
            continue

        b_obs = obs.reshape((-1, residual_policy.obs_dim))
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)

        next_value = residual_policy.get_value(next_residual_obs).reshape(1, -1).cpu()

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            steps_per_iteration,
            cfg.gamma,
            cfg.gae_lambda,
        )

        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            # if cfg.bc_coef > 0:
            #     demo_data_iter = iter(demo_data_loader)

            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                # Get the minibatch and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)

                # Calculate the loss
                _, newlogprob, entropy, newvalue, action_mean = (
                    residual_policy.get_action_and_value(mb_obs, mb_actions)
                )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                policy_loss = 0

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean() * cfg.ent_coef

                ppo_loss = pg_loss - entropy_loss
                # Add the auxiliary regularization loss
                aux_loss = torch.mean(torch.square(action_mean))

                # Normalize the losses so that each term has the same scale
                if iteration > cfg.n_iterations_train_only_value:
                    # Calculate the scaling factors based on the magnitudes of the losses
                    ppo_loss_scale = 1.0 / (torch.abs(ppo_loss.detach()) + 1e-8)
                    aux_loss_scale = 1.0 / (torch.abs(aux_loss.detach()) + 1e-8)

                    # Scale the losses using the calculated scaling factors
                    policy_loss += ppo_loss * ppo_loss_scale
                    policy_loss += (
                        cfg.residual_regularization * aux_loss * aux_loss_scale
                    )

                # Scale the value loss
                v_loss_scale = 1.0 / (torch.abs(v_loss.detach()) + 1e-8)
                scaled_v_loss = cfg.vf_coef * v_loss * v_loss_scale

                # Total loss
                loss = policy_loss + scaled_v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    residual_policy.parameters(), cfg.max_grad_norm
                )
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                print(
                    f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}"
                )
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        wandb.log(
            {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/total_loss": loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "losses/residual_l2": aux_loss.item(),
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "charts/rewards": rewards.sum().item(),
                "charts/success_rate": success_rate,
                "charts/action_norm_mean": action_norms.mean(),
                "charts/action_norm_std": action_norms.std(),
                "histograms/values": wandb.Histogram(values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/logprobs": wandb.Histogram(logprobs),
                "histograms/rewards": wandb.Histogram(rewards),
                "histograms/action_norms": wandb.Histogram(action_norms),
            },
            step=global_step,
        )

        # Checkpoint every cfg.checkpoint_interval steps
        if iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"iter_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": residual_policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "iteration": iteration,
                },
                model_path,
            )

            wandb.save(model_path)
            print(f"Model saved to {model_path}")

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()