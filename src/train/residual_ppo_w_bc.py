# import os
from pathlib import Path
import furniture_bench  # noqa

from ipdb import set_trace as bp


from src.common.hydra import to_native
from src.dataset.dataloader import FixedStepsDataloader
from src.dataset.rollout_buffer import RolloutBuffer
from tqdm import tqdm, trange
import random
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.dataset.dataset import StateDataset
from torch.utils.data import DataLoader

from src.common.pytorch_util import dict_to_device
from src.eval.eval_utils import get_model_from_api_or_cached
from diffusers.optimization import get_scheduler

from src.gym.env_rl_wrapper import RLPolicyEnvWrapper
from src.common.config_util import merge_base_bc_config_with_root_config

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv

from furniture_bench.envs.observation import DEFAULT_STATE_OBS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


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
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + discount * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


@hydra.main(
    config_path="../config",
    config_name="base_residual_rl",
    version_base="1.2",
)
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    # TRY NOT TO MODIFY: seeding
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    run_name = f"{int(time.time())}__residual_ppo__{cfg.actor.residual_policy._target_.split('.')[-1]}__{cfg.seed}"

    run_directory = f"runs/debug-residual_ppo-residual-8"
    run_directory += "-delete" if cfg.debug else ""
    print(f"Run directory: {run_directory}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}")

    turn_off_april_tags()

    env: FurnitureRLSimEnv = FurnitureRLSimEnv(
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=cfg.control.control_mode,
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode=cfg.control.controller,
        obs_keys=DEFAULT_STATE_OBS,
        furniture=cfg.env.task,
        # gpu_id=1,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space="state",
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000,
    )

    n_parts_to_assemble = len(env.pairs_to_assemble)

    # Load the behavior cloning actor
    base_cfg, base_wts = get_model_from_api_or_cached(
        cfg.base_policy.wandb_id,
        wt_type=cfg.base_policy.wt_type,
        wandb_mode=cfg.wandb.mode,
    )

    merge_base_bc_config_with_root_config(cfg, base_cfg)
    cfg.actor_name = f"residual_{cfg.base_policy.actor.name}"

    agent = ResidualDiffusionPolicy(device, base_cfg)
    agent.load_base_state_dict(base_wts)
    agent.to(device)
    agent.eval()

    residual_policy = agent.residual_policy

    # Set the inference steps of the actor
    if isinstance(agent, DiffusionPolicy):
        agent.inference_steps = 4

    env: RLPolicyEnvWrapper = RLPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        normalize_reward=cfg.normalize_reward,
        reset_on_success=cfg.reset_on_success,
        reset_on_failure=cfg.reset_on_failure,
        reward_clip=cfg.clip_reward,
        device=device,
    )

    optimizer_actor = optim.AdamW(
        agent.actor_parameters,
        lr=cfg.learning_rate_actor,
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    optimizer_critic = optim.AdamW(
        agent.critic_parameters,
        lr=cfg.learning_rate_critic,
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    optimizer_base = optim.AdamW(
        agent.base_actor_parameters,
        lr=cfg.base_bc.learning_rate,
        eps=1e-5,
        weight_decay=1e-6,
    )

    demo_dataset = StateDataset(
        dataset_paths=[Path(p) for p in to_native(base_cfg.data_path)],
        pred_horizon=cfg.data.pred_horizon,
        obs_horizon=cfg.data.obs_horizon,
        action_horizon=cfg.data.action_horizon,
        data_subset=cfg.data.data_subset,
        control_mode=cfg.control.control_mode,
        predict_past_actions=cfg.data.predict_past_actions,
        pad_after=cfg.data.get("pad_after", True),
        max_episode_count=cfg.data.get("max_episode_count", None),
        include_future_obs=cfg.data.include_future_obs,
    )

    # Create dataloaders
    demo_trainloader = FixedStepsDataloader(
        dataset=demo_dataset,
        batch_size=cfg.base_bc.batch_size,
        num_workers=0,
        # num_workers=cfg.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        n_batches=cfg.base_bc.max_updates_per_epoch,
    )

    steps_per_iteration = cfg.data_collection_steps

    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(
        f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}"
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "must",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
    )

    if cfg.wandb.continue_run_id is not None:
        print(f"Continuing run {cfg.wandb.continue_run_id}, {run.name}")

        run_id = f"{cfg.wandb.project}/{cfg.wandb.continue_run_id}"

        # Load the weights from the run
        _, wts = get_model_from_api_or_cached(
            run_id, "latest", wandb_mode=cfg.wandb.mode
        )

        print(f"Loading weights from {wts}")

        run_state_dict = torch.load(wts)

        if "actor_logstd" in run_state_dict["model_state_dict"]:
            agent.residual_policy.load_state_dict(run_state_dict["model_state_dict"])
        else:
            agent.load_state_dict(run_state_dict["model_state_dict"])

        optimizer_actor.load_state_dict(run_state_dict["optimizer_actor_state_dict"])
        optimizer_critic.load_state_dict(run_state_dict["optimizer_critic_state_dict"])
        lr_scheduler_actor.load_state_dict(run_state_dict["scheduler_actor_state_dict"])
        lr_scheduler_critic.load_state_dict(
            run_state_dict["scheduler_critic_state_dict"]
        )

        # Set the best test loss and success rate to the one from the run
        try:
            best_eval_success_rate = run.summary["eval/best_eval_success_rate"]
        except KeyError:
            best_eval_success_rate = run.summary["eval/success_rate"]

        iteration = run.summary["iteration"]
        global_step = run.step

    else:
        global_step = 0
        iteration = 0
        best_eval_success_rate = 0.0

    obs: torch.Tensor = torch.zeros(
        (
            steps_per_iteration,
            cfg.num_envs,
            residual_policy.obs_dim,
        )
    )
    actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape)
    full_nactions = torch.zeros(
        (steps_per_iteration, cfg.num_envs) + env.action_space.shape
    )
    logprobs = torch.zeros((steps_per_iteration, cfg.num_envs))
    rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    dones = torch.zeros((steps_per_iteration, cfg.num_envs))
    values = torch.zeros((steps_per_iteration, cfg.num_envs))

    start_time = time.time()
    training_cum_time = 0
    running_mean_success_rate = 0.0
    nominal_eval_performance = None

    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    agent.reset()

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # create replay buffer
    buffer = RolloutBuffer(
        max_size=cfg.base_bc.replay_buffer_size,
        state_dim=agent.obs_dim,
        action_dim=agent.action_dim,
        pred_horizon=agent.pred_horizon,
        obs_horizon=agent.obs_horizon,
        action_horizon=agent.action_horizon,
        device=device,
        predict_past_actions=cfg.data.predict_past_actions,
        include_future_obs=cfg.data.include_future_obs,
    )

    while global_step < cfg.total_timesteps:
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        iteration_start_time = time.time()

        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0

        # Also reset the env to have more consistent results
        if eval_mode or cfg.reset_every_iteration:
            next_obs = env.reset()
            agent.reset()

        print(f"Eval mode: {eval_mode}")

        for step in range(0, steps_per_iteration):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs

            # Get the base normalized action
            base_naction = agent.base_action_normalized(next_obs)

            # Process the obs for the residual policy
            next_nobs = agent.process_obs(next_obs)
            next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)

            dones[step] = next_done
            obs[step] = next_residual_nobs

            with torch.no_grad():
                residual_naction_samp, logprob, _, value, naction_mean = (
                    residual_policy.get_action_and_value(next_residual_nobs)
                )

            residual_naction = residual_naction_samp if not eval_mode else naction_mean
            naction = base_naction + residual_naction * residual_policy.action_scale

            action = agent.normalizer(naction, "action", forward=False)
            next_obs, reward, next_done, truncated, info = env.step(action)

            if cfg.truncation_as_done:
                next_done = next_done | truncated

            values[step] = value.flatten().cpu()
            actions[step] = residual_naction.cpu()
            logprobs[step] = logprob.cpu()
            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()
            full_nactions[step] = naction.cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()} fps={env_step * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
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

        running_mean_success_rate = 0.5 * running_mean_success_rate + 0.5 * success_rate

        print(
            f"SR: {success_rate:.4%}, SR mean: {running_mean_success_rate:.4%}, SPS: {steps_per_iteration * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
        )

        # We only want to store successful trajectories collected during evaluation
        # to not train on exploration noise
        if (
            eval_mode
            and nominal_eval_performance is not None
            and (success_rate - nominal_eval_performance)
            >= cfg.base_bc.improvement_threshold
        ):
            # Find which environments are successful, and fetch these trajectories
            success_idxs = rewards.sum(dim=0) >= n_parts_to_assemble
            success_obs = obs[:, success_idxs, :-10]
            success_actions = full_nactions[:, success_idxs]
            success_rewards = rewards[:, success_idxs]

            # This has all timesteps including and after episode is done
            success_dones = (
                rewards.cumsum(dim=0)[:, success_idxs] >= n_parts_to_assemble
            )

            # Let's mask out the ones that come after the first "done" was received
            first_done_mask = success_dones.cumsum(dim=0) > 1
            success_dones[first_done_mask] = False

            # Add the successful trajectories to the replay buffer
            buffer.add_trajectory(
                success_obs, success_actions, success_rewards, success_dones
            )

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue

            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(
                    {
                        # Save the weights of the residual policy (base + residual)
                        "model_state_dict": agent.state_dict(),
                        "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "success_timesteps_share": success_timesteps_share,
                        "iteration": iteration,
                    },
                    model_path,
                )

                wandb.save(model_path)
                print(f"Evaluation success rate improved. Model saved to {model_path}")

            wandb.log(
                {
                    "eval/success_rate": success_rate,
                    "eval/best_eval_success_rate": best_eval_success_rate,
                    "iteration": iteration,
                },
                step=global_step,
            )
            # Start the data collection again
            # NOTE: We're not resetting here now, that happens before the next
            # iteration only if the reset_every_iteration flag is set
            continue

        b_obs = obs.reshape((-1, residual_policy.obs_dim))
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)

        # Get the base normalized action
        # Process the obs for the residual policy
        base_naction = agent.base_action_normalized(next_obs)
        next_nobs = agent.process_obs(next_obs)
        next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)
        next_value = residual_policy.get_value(next_residual_nobs).reshape(1, -1).cpu()

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            next_done,
            steps_per_iteration,
            cfg.discount,
            cfg.gae_lambda,
        )

        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            early_stop = False

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
                residual_l1_loss = torch.mean(torch.abs(action_mean))
                residual_l2_loss = torch.mean(torch.square(action_mean))

                # Normalize the losses so that each term has the same scale
                if iteration > cfg.n_iterations_train_only_value:

                    # Scale the losses using the calculated scaling factors
                    policy_loss += ppo_loss
                    policy_loss += cfg.residual_l1 * residual_l1_loss
                    policy_loss += cfg.residual_l2 * residual_l2_loss

                # Total loss
                loss: torch.Tensor = policy_loss + v_loss * cfg.vf_coef

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_(
                    residual_policy.parameters(), cfg.max_grad_norm
                )

                optimizer_actor.step()
                optimizer_critic.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    print(
                        f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}"
                    )
                    early_stop = True
                    break

            if early_stop:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

        wandb.log(
            {
                "charts/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
                "charts/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
                "charts/SPS": sps,
                "charts/rewards": rewards.sum().item(),
                "charts/success_rate": success_rate,
                "charts/success_timesteps_share": success_timesteps_share,
                "charts/action_norm_mean": action_norms.mean(),
                "charts/action_norm_std": action_norms.std(),
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
                "losses/residual_l1": residual_l1_loss.item(),
                "losses/residual_l2": residual_l2_loss.item(),
                "histograms/values": wandb.Histogram(values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/logprobs": wandb.Histogram(logprobs),
                "histograms/rewards": wandb.Histogram(rewards),
                "histograms/action_norms": wandb.Histogram(action_norms),
            },
            step=global_step,
        )

        # Step the learning rate scheduler
        lr_scheduler_actor.step()
        lr_scheduler_critic.step()

        # Checkpoint every cfg.checkpoint_interval steps
        if iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"actor_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
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

        # Calculate how many training iterations we've done
        training_iterations = iteration - cfg.eval_first
        training_iterations -= (iteration - cfg.eval_first) // cfg.eval_interval

        print(
            f"At iteration {iteration}, we've done {training_iterations} training iterations "
            f"and {iteration - training_iterations} evaluation iterations"
        )

        if (
            cfg.base_bc.train_bc
            and buffer.size > 0
            and training_iterations % cfg.base_bc.train_with_bc_every == 0
        ):

            # Prepare the replay buffer and data loader for this epoch
            buffer.rebuild_seq_indices()
            buffer_trainloader = DataLoader(
                buffer,
                batch_size=cfg.base_bc.batch_size,
                # num_workers=cfg.data.dataloader_workers,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
            )

            demo_epoch_loss = list()
            buffer_epoch_loss = list()
            grad_norms = list()
            gradient_steps = 0

            agent.train()

            for epoch in range(cfg.base_bc.num_epochs):

                tepoch = tqdm(
                    zip(demo_trainloader, buffer_trainloader),
                    desc="Training",
                    leave=True,
                    total=min(len(demo_trainloader), len(buffer_trainloader)),
                )
                # Train the base policy with BC for a few iterations
                for demo_batch, buffer_batch in tepoch:

                    # Zero the gradients in all optimizers
                    optimizer_base.zero_grad()

                    # Make predictions with agent
                    demo_batch = dict_to_device(demo_batch, device)
                    demo_loss, _ = agent.compute_loss(demo_batch, base_only=True)
                    (demo_loss / 2).backward()

                    # Make predictions with agent
                    buffer_batch = dict_to_device(buffer_batch, device)
                    buffer_loss, _ = agent.compute_loss(buffer_batch, base_only=True)
                    (buffer_loss / 2).backward()

                    # Clip to a big value if we don't want to clip the gradients
                    # so that we can still log the gradient norms
                    grad_norm = nn.utils.clip_grad_norm_(
                        agent.base_actor_parameters,
                        1 if cfg.base_bc.clip_grad_norm else 1000,
                    ).item()
                    grad_norms.append(grad_norm)

                    # Step the optimizers and schedulers
                    optimizer_base.step()
                    gradient_steps += 1

                    # Log losses
                    demo_loss_cpu = demo_loss.item()
                    buffer_loss_cpu = buffer_loss.item()

                    demo_epoch_loss.append(demo_loss_cpu)
                    buffer_epoch_loss.append(buffer_loss_cpu)

                    tepoch.set_postfix(
                        base_loss=demo_loss_cpu, buffer_loss=buffer_loss_cpu
                    )

            mean_demo_loss = np.mean(demo_epoch_loss)
            mean_buffer_loss = np.mean(buffer_epoch_loss)
            mean_loss = (mean_demo_loss + mean_buffer_loss) / 2

            wandb.log(
                {
                    "base_bc/demo_loss": mean_demo_loss,
                    "base_bc/buffer_loss": mean_buffer_loss,
                    "base_bc/loss": mean_loss,
                    "base_bc/grad_norm": np.mean(grad_norms),
                    "base_bc/epoch": training_iterations,
                    "base_bc/iteration": iteration,
                    "base_bc/gradient_steps": gradient_steps,
                },
                step=global_step,
            )
            agent.eval()

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()
