# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from pathlib import Path

import random
import time
import math

import hydra
from omegaconf import DictConfig, OmegaConf
from src.common.config_util import merge_base_bc_config_with_root_config
from src.common.files import get_processed_paths
from src.eval.eval_utils import get_model_from_api_or_cached
from src.gym.env_rl_wrapper import FurnitureEnvRLWrapper
import numpy as np
from src.common.pytorch_util import dict_to_device
from src.dataset.dataset import StateDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from src.dataset.normalizer import LinearNormalizer
from src.gym.observation import DEFAULT_STATE_OBS


from src.behavior.mlp import MLPActor

from ipdb import set_trace as bp

# Set the gym logger to not print to console
import gym

gym.logger.set_level(40)
import wandb
import copy
from torch.distributions.normal import Normal


from src.gym import get_rl_env
import gymnasium as gym

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


def get_demo_data_loader(
    control_mode,
    n_batches,
    num_workers=4,
    action_horizon=1,
    add_relative_pose=False,
    normalizer=None,
    task="one_leg",
    controller="diffik",
) -> DataLoader:

    paths = get_processed_paths(
        controller=controller,
        domain="sim",
        task=task,
        demo_source="teleop",
        randomness="low",
        demo_outcome="success",
    )

    demo_data = StateDataset(
        dataset_paths=paths,
        obs_horizon=1,
        pred_horizon=action_horizon,
        action_horizon=action_horizon,
        data_subset=None,
        control_mode=control_mode,
        pad_after=False,
        predict_past_actions=True,
        max_episode_count=None,
        add_relative_pose=add_relative_pose,
        normalizer=normalizer,
    )

    batch_size = len(demo_data) // n_batches

    demo_data_loader = DataLoader(
        dataset=demo_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    return demo_data_loader


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


@hydra.main(config_path="../config", config_name="base_mlp_ppo", version_base="1.2")
def main(cfg: DictConfig):

    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    run_name = f"{int(time.time())}__mlp_ppo__{cfg.seed}"

    run_directory = f"runs/mlp-ppo"
    run_directory += "-debug" if cfg.debug else ""
    print(f"Run directory: {run_directory}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    gpu_id = cfg.gpu_id
    device = torch.device(f"cuda:{gpu_id}")

    # Load the behavior cloning actor
    base_cfg, base_wts = get_model_from_api_or_cached(
        cfg.base_policy.wandb_id,
        wt_type=cfg.base_policy.wt_type,
        wandb_mode=cfg.wandb.mode,
    )

    merge_base_bc_config_with_root_config(cfg, base_cfg)
    cfg.actor_name = cfg.base_policy.actor.name

    agent = MLPActor(device, base_cfg)
    agent.load_bc_weights(base_wts)
    agent.to(device)
    agent.eval()

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

    n_parts_to_assemble = env.n_parts_assemble

    env: FurnitureEnvRLWrapper = FurnitureEnvRLWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        chunk_size=agent.action_horizon,
        normalize_reward=cfg.normalize_reward,
        reward_clip=cfg.clip_reward,
    )

    normalizer = LinearNormalizer()
    print(normalizer.load_state_dict(agent.normalizer.state_dict()))
    env.set_normalizer(normalizer)

    if cfg.kl_coef > 0:
        ref_agent = MLPActor(device, base_cfg)
        ref_agent.load_state_dict(copy.deepcopy(agent.state_dict()))
        ref_agent = ref_agent.to(device)

    optimizer = optim.AdamW(
        agent.parameters(),
        lr=cfg.learning_rate,
        eps=1e-5,
        # weight_decay=1e-5,
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

    best_eval_success_rate = 0.0

    obs = torch.zeros((num_steps, cfg.num_envs) + env.observation_space.shape)
    actions = torch.zeros((num_steps, cfg.num_envs) + env.action_space.shape)
    logprobs = torch.zeros((num_steps, cfg.num_envs))
    rewards = torch.zeros((num_steps, cfg.num_envs))
    dones = torch.zeros((num_steps, cfg.num_envs))
    values = torch.zeros((num_steps, cfg.num_envs))

    # Get the dataloader for the demo data for the behavior cloning
    demo_data_loader = get_demo_data_loader(
        control_mode=cfg.control.control_mode,
        n_batches=cfg.num_minibatches,
        normalizer=normalizer,
        action_horizon=agent.action_horizon,
        num_workers=4 if not cfg.debug else 0,
        task=cfg.env.task,
        controller=cfg.control.controller,
    )

    # Print the number of batches in the dataloader
    print(f"Number of batches in the dataloader: {len(demo_data_loader)}")

    agent = agent.to(device)

    # Number of environment steps
    global_step = 0
    training_cum_time = 0

    start_time = time.time()

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

            with torch.no_grad():
                naction, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten().cpu()

            # Clamp action to be [-5, 5], arbitrary value
            naction = torch.clamp(naction, -5, 5)

            next_obs, reward, next_done, infos = env.step(naction)

            actions[step] = naction.cpu()
            logprobs[step] = logprob.cpu()

            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step) % (100 // agent.action_horizon) == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()}"
                )
        # Calculate the success rate
        # Find the rewards that are not zero
        # Env is successful if it received a reward more than or equal to n_parts_to_assemble
        env_success = (rewards > 0).sum(dim=0) >= n_parts_to_assemble
        success_rate = env_success.float().mean().item()

        # Calculate the share of timesteps that come from successful trajectories that account for the success rate and the varying number of timesteps per trajectory
        # Count total timesteps in successful trajectories
        timesteps_in_success = rewards[:, env_success]

        # Find index of last reward in each trajectory
        last_reward_idx = torch.argmax(timesteps_in_success, dim=0)

        # Calculate the total number of timesteps in successful trajectories
        total_timesteps_in_success = last_reward_idx.sum().item()

        # Calculate the share of successful timesteps
        success_timesteps_share = total_timesteps_in_success / rewards.numel()

        print(
            f"SR: {success_rate:.4%}, SPS: {cfg.num_env_steps * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
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
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)

        # bootstrap value if not done
        next_value = agent.get_value(next_obs).reshape(1, -1).cpu()

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
        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()

        agent.train()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            if cfg.bc_coef > 0:
                demo_data_iter = iter(demo_data_loader)

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
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, mb_actions
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

                if iteration > cfg.n_iterations_train_only_value:
                    policy_loss += ppo_loss

                # Behavior cloning loss
                if cfg.bc_coef > 0:
                    batch = next(demo_data_iter)
                    batch = dict_to_device(batch, device)
                    bc_obs = batch["obs"].squeeze()

                    norm_bc_actions = batch["action"]

                    # Normalize the actions
                    if cfg.bc_loss_type == "mse":
                        action_pred = agent.actor_mean(bc_obs)
                        bc_loss = F.mse_loss(action_pred, norm_bc_actions)
                    elif cfg.bc_loss_type == "nll":
                        _, bc_logprob, _, bc_values = agent.get_action_and_value(
                            bc_obs, norm_bc_actions
                        )
                        bc_loss = -bc_logprob.mean()
                    else:
                        raise ValueError(
                            f"Unknown behavior cloning loss type: {cfg.bc_loss_type}"
                        )

                    with torch.no_grad():
                        bc_values = agent.get_value(bc_obs)
                    bc_v_loss = (
                        0.5
                        * (
                            (bc_values.squeeze() - batch["returns"].squeeze()) ** 2
                        ).mean()
                    )

                    bc_total_loss = (
                        bc_loss + bc_v_loss if cfg.supervise_value_function else bc_loss
                    )

                    policy_loss = (
                        1 - cfg.bc_coef
                    ) * policy_loss + cfg.bc_coef * bc_total_loss

                # KL regularization loss
                if cfg.kl_coef > 0:
                    mb_new_actions, _, _, _ = agent.get_action_and_value(mb_obs)
                    with torch.no_grad():
                        action_mean = ref_agent.actor_mean(mb_obs)
                        action_logstd = ref_agent.actor_logstd.expand_as(action_mean)
                        action_std = torch.exp(action_logstd)
                        ref_dist = Normal(action_mean, action_std)
                    kl_loss = -ref_dist.log_prob(mb_new_actions).mean()
                    policy_loss = policy_loss + cfg.kl_coef * kl_loss
                # Total loss
                loss = policy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if (
                cfg.target_kl is not None
                and approx_kl > cfg.target_kl
                and cfg.bc_coef < 1.0
            ):
                print(
                    f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}"
                )
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        advantages = advantages.cpu()
        returns = returns.cpu()

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

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
                "losses/policy_loss": pg_loss.item(),
                "losses/total_loss": loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "histograms/values": wandb.Histogram(values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/logprobs": wandb.Histogram(logprobs),
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
