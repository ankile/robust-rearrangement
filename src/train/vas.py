# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from pathlib import Path

import random
import time
import math

import hydra
from omegaconf import DictConfig, OmegaConf
from src.common.config_util import merge_base_bc_config_with_root_config
from src.eval.eval_utils import get_model_from_api_or_cached
from src.gym.env_rl_wrapper import FurnitureEnvRLWrapper
from src.gym.observation import DEFAULT_STATE_OBS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from src.dataset.normalizer import LinearNormalizer


from ipdb import set_trace as bp

# Set the gym logger to not print to console
import gymnasium as gym

gym.logger.set_level(40)

import wandb

from src.behavior.diffusion import DiffusionPolicy

from src.gym import get_rl_env


# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


def get_model_weights(run_id: str):
    api = wandb.Api()
    run = api.run(run_id)
    model_path = (
        [
            f
            for f in run.files()
            if f.name.endswith(".pt") and "best_test_loss" in f.name
        ][0]
        .download(exist_ok=True)
        .name
    )
    print(f"Loading checkpoint from {run_id}")
    return torch.load(model_path)


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    steps_per_iteration = values.size(0)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_zeros(layer, bias_const=0.0):
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Qfunction(nn.Module):
    def __init__(self, obs_shape, action_shape):
        """
        Args:
            obs_shape: the shape of the observation (i.e., state + base action)
            action_shape: the shape of the action (i.e., residual, same size as base action)
        """
        super().__init__()

        self.action_horizon, self.action_dim = action_shape
        self.obs_dim = np.prod(obs_shape) + np.prod(action_shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init_zeros(nn.Linear(512, 1)),
            nn.Sigmoid(),
        )

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)


@hydra.main(config_path="../config", config_name="base_vas", version_base="1.2")
def main(cfg: DictConfig):
    sigma = float(cfg.sigma)

    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    run_name = f"{int(time.time())}__{cfg.actor.name}_vas__{cfg.seed}"

    run_directory = f"runs/mlp-vas"
    run_directory += "-delete" if cfg.debug else ""
    print(f"Run directory: {run_directory}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda")

    # Load the behavior cloning actor
    base_cfg, base_wts = get_model_from_api_or_cached(
        cfg.base_policy.wandb_id,
        wt_type=cfg.base_policy.wt_type,
        wandb_mode=cfg.wandb.mode,
    )

    merge_base_bc_config_with_root_config(cfg, base_cfg)
    cfg.actor_name = cfg.base_policy.actor.name
    # base_cfg.actor.dropout = 0.0
    # base_cfg.actor.critic.last_layer_bias_const = 0.5

    agent = DiffusionPolicy(device, base_cfg)
    # Set the inference steps of the actor
    if isinstance(agent, DiffusionPolicy):
        agent.inference_steps = 4

    agent.eta = cfg.eta

    base_state_dict = torch.load(base_wts)

    if "model_state_dict" in base_state_dict:
        base_state_dict = base_state_dict["model_state_dict"]

    # Load the model weights
    agent.load_state_dict(base_state_dict)
    # agent.eta = 1.0
    agent.to(device)
    agent.eval()

    Q_estimator = Qfunction(
        base_cfg.timestep_obs_dim, (base_cfg.action_horizon, base_cfg.action_dim)
    )
    Q_estimator.to(device)
    Q_estimator.train()

    gpu_id = cfg.gpu_id
    device = torch.device(f"cuda:{gpu_id}")

    # env setup
    env: gym.Env = get_rl_env(
        gpu_id=gpu_id,
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=cfg.control.control_mode,
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode=cfg.control.controller,
        obs_keys=DEFAULT_STATE_OBS,
        task=cfg.env.task,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space="state",
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000,
    )

    env: FurnitureEnvRLWrapper = FurnitureEnvRLWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        chunk_size=agent.action_horizon,
        reset_on_success=cfg.reset_on_success,
    )

    normalizer = LinearNormalizer()
    print(normalizer.load_state_dict(agent.normalizer.state_dict()))
    env.set_normalizer(normalizer)

    optimizer = optim.AdamW(
        Q_estimator.parameters(),
        lr=cfg.learning_rate,
        eps=1e-5,
        weight_decay=1e-5,
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
    )

    policy_steps = math.ceil(cfg.num_env_steps / agent.action_horizon)
    num_steps = cfg.data_collection_steps
    cfg.batch_size = int(cfg.num_envs * num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size
    print(
        f"With chunk size {agent.action_horizon}, we have {policy_steps} policy steps."
    )
    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(
        f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}"
    )

    running_mean_success_rate = 0.0
    best_eval_success_rate = 0.0

    obs = torch.zeros((num_steps, cfg.num_envs) + env.observation_space.shape)
    actions = torch.zeros((num_steps, cfg.num_envs) + env.action_space.shape)
    rewards = torch.zeros((num_steps, cfg.num_envs))
    dones = torch.zeros((num_steps, cfg.num_envs))
    values = torch.zeros((num_steps, cfg.num_envs))

    agent = agent.to(device)
    global_step = 0
    training_cum_time = 0

    start_time = time.time()
    # bp()
    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, cfg.num_iterations + 1):
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        agent.eval()
        iteration_start_time = time.time()

        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0

        # Also reset the env to have more consistent results
        if eval_mode or cfg.reset_every_iteration:
            next_obs = env.reset()

        print(f"Eval mode: {eval_mode}")

        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs * agent.action_horizon

            obs[step] = next_obs
            dones[step] = next_done

            if not eval_mode:
                with torch.no_grad():
                    out = agent._normalized_action(next_obs)
                    naction = out[:, : agent.action_horizon, :]
                    naction = naction + sigma * torch.randn(
                        naction.shape,
                        device=naction.device,
                    )
                    critic_obs = torch.cat(
                        [next_obs, naction.reshape(cfg.num_envs, -1)], dim=-1
                    )
                    values[step] = Q_estimator.get_value(critic_obs).squeeze()

            if eval_mode:
                with torch.no_grad():
                    k = 20
                    expanded_next_obs = next_obs.unsqueeze(1).repeat(1, k, 1)
                    expanded_next_obs = expanded_next_obs.reshape(cfg.num_envs * k, -1)
                    out = agent._normalized_action(expanded_next_obs)
                    naction = out[:, : agent.action_horizon, :]
                    naction = naction + sigma * torch.randn(
                        naction.shape,
                        device=naction.device,
                    )
                    # t = naction[:, 0, :3].view(cfg.num_envs, k, -1).cpu().numpy()
                    # var = np.var(t, axis=1).mean(0)
                    critic_obs = torch.cat(
                        [expanded_next_obs, naction.reshape(cfg.num_envs * k, -1)],
                        dim=-1,
                    )
                    Qvalues = Q_estimator.get_value(critic_obs).squeeze()
                    Qvalues = Qvalues.reshape(cfg.num_envs, k, -1)
                    indices = torch.argmax(Qvalues, dim=1).squeeze()
                    naction = naction.reshape(cfg.num_envs, k, agent.action_horizon, -1)
                    naction = naction[torch.arange(cfg.num_envs), indices, :, :]

            # Clamp action to be [-5, 5], arbitrary value
            naction = torch.clamp(naction, -5, 5)

            next_obs, reward, next_done, infos = env.step(naction)

            actions[step] = naction.cpu()

            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step) % (100 // agent.action_horizon) == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()}"
                )

        # Calculate the success rate
        reward_mask = rewards.sum(dim=0) > 0
        success_rate = reward_mask.float().mean().item()

        # Calculate the share of timesteps that come from successful trajectories that account for the success rate and the varying number of timesteps per trajectory
        # Count total timesteps in successful trajectories
        timesteps_in_success = rewards[:, reward_mask]

        # Find index of last reward in each trajectory
        last_reward_idx = torch.argmax(timesteps_in_success, dim=0)

        # Calculate the total number of timesteps in successful trajectories
        total_timesteps_in_success = last_reward_idx.sum().item()

        # Calculate the share of successful timesteps
        success_timesteps_share = total_timesteps_in_success / rewards.numel()

        running_mean_success_rate = 0.5 * running_mean_success_rate + 0.5 * success_rate

        print(
            f"SR: {success_rate:.4%}, SR mean: {running_mean_success_rate:.4%}, SPS: {cfg.num_env_steps * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
        )

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue
            wandb.log(
                {"eval/success_rate": success_rate, "iteration": iteration},
                step=global_step,
            )

            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")

                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "optimizer_actor_state_dict": optimizer.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "iteration": iteration,
                    },
                    model_path,
                )

                wandb.save(model_path)
                print(f"Evaluation success rate improved. Model saved to {model_path}")

            # Start the data collection again
            # NOTE: We're not resetting here now, that happens before the next
            # iteration only if the reset_every_iteration flag is set
            continue

        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_values = values.reshape(-1)

        # bootstrap value if not done
        with torch.no_grad():
            out = agent._normalized_action(next_obs)
            next_naction = out[:, : agent.action_horizon, :]
            next_naction = next_naction + sigma * torch.randn(
                next_naction.shape,
                device=next_naction.device,
            )
            next_critic_obs = torch.cat(
                [next_obs, next_naction.reshape(cfg.num_envs, -1)], dim=-1
            )
            next_value = Q_estimator.get_value(next_critic_obs).reshape(1, -1)
            next_value = next_value.reshape(1, -1).cpu()

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            next_done,
            cfg.gamma,
            cfg.gae_lambda,
        )
        mask = torch.ones_like(returns)
        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()
        b_mask = mask.reshape(-1).cpu()

        agent.train()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        for epoch in trange(cfg.update_epochs, desc="Policy update"):

            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]
                curr_mb_size = mb_inds.shape[0]

                # Get the minibatch and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)
                mb_mask = b_mask[mb_inds].to(device)

                # Calculate the loss
                mb_critic_obs = torch.cat(
                    [mb_obs, mb_actions.reshape(curr_mb_size, -1)], dim=-1
                )
                newvalue = Q_estimator.get_value(mb_critic_obs)

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
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2)
                    v_loss = (v_loss * mb_mask).sum() / mb_mask.sum()

                # Total loss
                loss: torch.Tensor = cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    Q_estimator.parameters(), cfg.max_grad_norm
                )
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

        # bp()

        # Check if any of the variables passed to the histograms are NaN
        if any(
            torch.isnan(x).any() for x in [b_values, b_returns, b_advantages, rewards]
        ):
            print("NaN detected in the variables passed to the histograms")
            # bp()

        wandb.log(
            {
                "charts/SPS": sps,
                "charts/rewards": rewards.sum().item(),
                "charts/success_rate": success_rate,
                "charts/success_timesteps_share": success_timesteps_share,
                "values/advantages": b_advantages.mean().item(),
                "values/returns": b_returns.mean().item(),
                "values/values": b_values.mean().item(),
                "losses/value_loss": v_loss.item(),
                "losses/total_loss": loss.item(),
                "losses/explained_variance": explained_var,
                "losses/grad_norm": grad_norm,
                "histograms/values": wandb.Histogram(b_values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/rewards": wandb.Histogram(rewards),
            },
            step=global_step,
        )

        # Checkpoint every cfg.checkpoint_interval steps
        if iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"actor_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_actor_state_dict": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "iteration": iteration,
                },
                model_path,
            )

            wandb.save(model_path)
            print(f"Model saved to {model_path}")

        # Print some stats at the end of the iteration
        print(
            f"Iteration {iteration}/{cfg.num_iterations}, global step {global_step}, SPS {sps}"
        )

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()
