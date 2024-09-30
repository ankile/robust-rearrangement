# import os
from pathlib import Path
import furniture_bench  # noqa

from furniture_bench.controllers.control_utils import proprioceptive_quat_to_6d_rotation
from ipdb import set_trace as bp


from src.common.hydra import to_native
from src.dataset.normalizer import LinearNormalizer
from src.models.residual import ResidualPolicy
from tqdm import tqdm, trange
import random
import time

# from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf

from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from torch.utils.data import DataLoader

# from src.dataset.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_to_device
from src.eval.eval_utils import get_model_from_api_or_cached
from src.common.cosine_annealing_warmup import CosineAnnealingWarmupRestarts


from src.gym.env_rl_wrapper import RLPolicyEnvWrapper

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv

from furniture_bench.envs.observation import (
    DEFAULT_STATE_OBS,
    DEFAULT_VISUAL_OBS,
)
from src.data_processing.utils import resize, resize_crop
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from src.gym import turn_off_april_tags

from src.dataset.rollout_buffer import RolloutBuffer
from src.common.config_util import merge_student_config_with_root_config

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


def resize_image(obs, key):
    try:
        obs[key] = resize(obs[key])
    except KeyError:
        pass


def resize_crop_image(obs, key):
    try:
        obs[key] = resize_crop(obs[key])
    except KeyError:
        pass


@hydra.main(
    config_path="../config",
    config_name="base_dagger",
    version_base="1.2",
)
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    # TRY NOT TO MODIFY: seeding
    if cfg.seed is None:
        cfg.seed = random.randint(0, 2**32 - 1)

    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    run_name = f"{int(time.time())}__dagger__{cfg.seed}"

    run_directory = f"runs/dagger"
    run_directory += "-debug" if cfg.debug else ""
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
        obs_keys=(
            (DEFAULT_VISUAL_OBS + ["parts_poses"])
            if cfg.observation_type == "image"
            else DEFAULT_STATE_OBS
        ),
        furniture=cfg.env.task,
        # gpu_id=1,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space=cfg.observation_type,  # "state",
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000,
        resize_img=False,
    )
    n_parts_to_assemble = len(env.pairs_to_assemble)

    env: RLPolicyEnvWrapper = RLPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        normalize_reward=False,
        reset_on_success=False,
        reset_on_failure=False,
        device=device,
    )

    # Load the teacher policy
    teacher_cfg, teacher_wts_path = get_model_from_api_or_cached(
        cfg.teacher_policy.wandb_id,
        wt_type=cfg.teacher_policy.wt_type,
        wandb_mode=cfg.wandb.mode,
    )

    # Update the teacher_policy config with the one from the run
    cfg.teacher_policy.merge_with(teacher_cfg)

    teacher = ResidualDiffusionPolicy(device, teacher_cfg)

    teacher_wts = torch.load(teacher_wts_path)
    if "model_state_dict" in teacher_wts:
        teacher_wts = teacher_wts["model_state_dict"]

    teacher.load_state_dict(teacher_wts)
    teacher.to(device)
    teacher.eval()

    # Load student policy
    student_cfg, student_wts_path = get_model_from_api_or_cached(
        cfg.student_policy.wandb_id,
        wt_type=cfg.student_policy.wt_type,
        wandb_mode=cfg.wandb.mode,
    )

    # Update the student_policy config with the one from the run
    # cfg.student_policy.merge_with(student_cfg)
    merge_student_config_with_root_config(cfg, student_cfg)
    student = DiffusionPolicy(device, student_cfg)
    if student_wts_path is not None:
        student_wts = torch.load(student_wts_path)
        if "model_state_dict" in student_wts:
            student_wts = student_wts["model_state_dict"]

        student.load_state_dict(student_wts)

    else:
        student.normalizer.load_state_dict(teacher.normalizer.state_dict())

    normalizer = LinearNormalizer()

    # We'll use this normalizer to normalize the observations and actions
    # before putting them in the training dataset
    normalizer.load_state_dict(student.normalizer.state_dict())
    normalizer.cpu()
    student.to(device)
    student.eval()

    optimizer_student = optim.AdamW(
        student.parameters(),
        lr=cfg.learning_rate_student,
        eps=1e-5,
        weight_decay=1e-6,
    )

    assert cfg.lr_scheduler.name == "cosine"
    lr_scheduler_student = CosineAnnealingWarmupRestarts(
        optimizer=optimizer_student,
        first_cycle_steps=cfg.num_iterations,
        cycle_mult=1.0,
        max_lr=cfg.learning_rate_student,
        warmup_steps=cfg.lr_scheduler.warmup_steps,
        min_lr=cfg.lr_scheduler.min_lr,
    )

    steps_per_iteration = cfg.num_env_steps

    print(f"Num iterations: {cfg.num_iterations}, batch size: {cfg.batch_size}")

    print(OmegaConf.to_yaml(cfg, resolve=True))

    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
    )

    if cfg.wandb.continue_run_id is not None and run.step > 0:
        print(f"Continuing run {cfg.wandb.continue_run_id}, {run.name}")

        run_id = f"{cfg.wandb.project}/{cfg.wandb.continue_run_id}"

        # Load the weights from the run
        _, wts = get_model_from_api_or_cached(
            run_id, "latest", wandb_mode=cfg.wandb.mode
        )

        print(f"Loading weights from {wts}")

        run_state_dict = torch.load(wts)
        student.load_state_dict(run_state_dict["model_state_dict"])

        optimizer_student.load_state_dict(
            run_state_dict["optimizer_student_state_dict"]
        )
        lr_scheduler_student.load_state_dict(
            run_state_dict["scheduler_student_state_dict"]
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

    robot_states: torch.Tensor = torch.zeros(
        (
            steps_per_iteration,
            cfg.num_envs,
            env.env.observation_space.spaces["robot_state"].shape[0],
        )
    )
    parts_poses: torch.Tensor = torch.zeros(
        (
            steps_per_iteration,
            cfg.num_envs,
            env.env.observation_space.spaces["parts_poses"].shape[0],
        )
    )
    actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape)
    rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    dones = torch.zeros((steps_per_iteration, cfg.num_envs))
    color_images1: torch.Tensor = torch.zeros(
        (steps_per_iteration, cfg.num_envs, 240, 320, 3), dtype=torch.uint8
    )
    color_images2: torch.Tensor = torch.zeros(
        (steps_per_iteration, cfg.num_envs, 240, 320, 3), dtype=torch.uint8
    )
    is_action_from_student = torch.zeros((steps_per_iteration, cfg.num_envs))

    start_time = time.time()
    training_cum_time = 0

    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    student.reset()

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # create replay buffer
    robot_state_dim_6d = env.env.observation_space["robot_state"].shape[0] + 2
    parts_poses_dim = env.env.observation_space["parts_poses"].shape[0]
    buffer = RolloutBuffer(
        max_size=cfg.replay_buffer_size,
        state_dim=robot_state_dim_6d + parts_poses_dim,
        action_dim=student.action_dim,
        pred_horizon=student.pred_horizon,
        obs_horizon=student.obs_horizon,
        action_horizon=student.action_horizon,
        device=device,
        predict_past_actions=cfg.student_policy.data.predict_past_actions,
        include_future_obs=cfg.student_policy.data.include_future_obs,
        include_images=(cfg.observation_type == "image"),
    )

    beta = cfg.beta
    last_success_rate = 0.0
    reference_success_rate = None
    gradient_steps = 0
    training_samples = 0

    while iteration < cfg.num_iterations:
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        iteration_start_time = time.time()

        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0
        next_obs = env.reset()
        student.reset()
        teacher.reset()

        print(f"Eval mode: {eval_mode}")
        if reference_success_rate is not None and last_success_rate > (
            cfg.beta_decay_ref_sr_ratio * reference_success_rate
        ):
            beta = max(cfg.beta_min, beta - cfg.beta_linear_decay)
            print(
                f"Reference success rate: {reference_success_rate}, last success rate: {last_success_rate}, new beta: {beta}"
            )

        for step in range(0, steps_per_iteration):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs

            dones[step] = next_done
            robot_states[step] = next_obs["robot_state"]
            parts_poses[step] = next_obs["parts_poses"]

            if buffer.include_images:
                next_obs["color_image1"] = resize(next_obs["color_image1"])
                next_obs["color_image2"] = resize_crop(next_obs["color_image2"])
                color_images1[step] = next_obs["color_image1"]
                color_images2[step] = next_obs["color_image2"]

            with torch.no_grad():
                # Get the actions from the student and the teacher
                student_action = student.action(next_obs)
                if cfg.correct_student_action_only:
                    teacher_action = teacher.correct_action(next_obs, student_action)
                else:
                    teacher_action = teacher.action(next_obs)

            # Always use the student action during evaluation
            # Otherwise, use the teacher action with probability beta
            beta_to_use = beta if iteration > cfg.teacher_only_iters else 1.0
            is_student_action = torch.full(
                (cfg.num_envs, 1), eval_mode, device=device, dtype=torch.bool
            ) | (torch.rand(cfg.num_envs, 1, device=device) > beta_to_use)
            is_action_from_student[step] = is_student_action.view(-1)

            action = torch.where(
                is_student_action,
                student_action,
                teacher_action,
            )

            next_obs, reward, next_done, truncated, info = env.step(action)

            if cfg.truncation_as_done:
                next_done = next_done | truncated

            actions[step] = teacher_action.cpu()
            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()} fps={env_step * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
                )

        # Calculate the success rate
        # Find the rewards that are not zero
        # Env is successful if it received a reward more than or equal to n_parts_to_assemble
        env_success = (rewards > 0).sum(dim=0) >= n_parts_to_assemble
        success_rate = env_success.float().mean().item()
        if iteration == 1:
            reference_success_rate = success_rate
        else:
            last_success_rate = success_rate

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
            f"SR: {success_rate:.4%}, SPS: {steps_per_iteration * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
        )

        # Evaluation use only student actions so we don't want to store those
        if not eval_mode:
            # Find which environments are successful, and fetch these trajectories
            success_idxs = rewards.sum(dim=0) >= n_parts_to_assemble

            success_robot_states = robot_states[:, success_idxs]
            success_parts_poses = parts_poses[:, success_idxs]
            success_actions = actions[:, success_idxs]
            success_rewards = rewards[:, success_idxs]

            # This has all timesteps including and after episode is done
            success_dones = (
                rewards.cumsum(dim=0)[:, success_idxs] >= n_parts_to_assemble
            )

            # Let's mask out the ones that come after the first "done" was received
            first_done_mask = success_dones.cumsum(dim=0) > 1
            success_dones[first_done_mask] = False

            success_robot_states = proprioceptive_quat_to_6d_rotation(
                success_robot_states
            )

            success_actions = normalizer(success_actions, "action", forward=True)

            if not buffer.include_images:
                # Normalize the observations and actions so we only do this once
                success_obs = torch.cat(
                    [
                        normalizer(success_robot_states, "robot_state", forward=True),
                        normalizer(success_parts_poses, "parts_poses", forward=True),
                    ],
                    dim=-1,
                )
                # Add the successful trajectories to the replay buffer
                buffer.add_trajectories(
                    states=success_obs,
                    actions=success_actions,
                    rewards=success_rewards,
                    dones=success_dones,
                )
            else:
                # Add the successful trajectories to the replay buffer
                success_robot_states = normalizer(
                    success_robot_states, "robot_state", forward=True
                )
                success_color_images1 = color_images1[:, success_idxs]
                success_color_images2 = color_images2[:, success_idxs]
                buffer.add_trajectories(
                    actions=success_actions,
                    rewards=success_rewards,
                    dones=success_dones,
                    robot_states=success_robot_states,
                    color_images1=success_color_images1,
                    color_images2=success_color_images2,
                )

        # Checkpoint every cfg.checkpoint_interval steps
        if iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"student_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "optimizer_student_state_dict": optimizer_student.state_dict(),
                    "scheduler_student_state_dict": lr_scheduler_student.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "iteration": iteration,
                },
                model_path,
            )

            wandb.save(model_path)
            print(f"Model saved to {model_path}")

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue

            # Save the model if the evaluation success rate improves
            if success_rate >= best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"student_chkpt_best_success_rate.pt")
                torch.save(
                    {
                        # Save the weights of the residual policy (base + residual)
                        "model_state_dict": student.state_dict(),
                        "optimizer_student_state_dict": optimizer_student.state_dict(),
                        "scheduler_student_state_dict": lr_scheduler_student.state_dict(),
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

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

        wandb.log(
            {
                "run_stats/learning_rate_student": optimizer_student.param_groups[0][
                    "lr"
                ],
                "run_stats/SPS": sps,
                "charts/rewards": rewards.sum().item(),
                "charts/success_rate": success_rate,
                "charts/success_timesteps_share": success_timesteps_share,
                "histograms/rewards": wandb.Histogram(rewards),
            },
            step=global_step,
        )

        # Prepare the replay buffer and data loader for this epoch
        buffer.rebuild_seq_indices()
        trainloader = DataLoader(
            buffer,
            batch_size=cfg.batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )

        demo_epoch_loss = list()
        grad_norms = list()

        # Calculate how many training iterations we've done
        training_iterations = iteration - cfg.eval_first
        training_iterations -= (iteration - cfg.eval_first) // cfg.eval_interval

        student.train()
        for epoch in range(cfg.num_epochs):

            tepoch = tqdm(
                zip(trainloader, range(cfg.max_steps_per_epoch)),
                desc=f"Epoch {epoch}/{cfg.num_epochs}",
                total=min(len(trainloader), cfg.max_steps_per_epoch),
            )
            # Train the base policy with BC for a few iterations
            for batch, _ in tepoch:

                # Zero the gradients in all optimizers
                optimizer_student.zero_grad()

                # Make predictions with agent
                batch = dict_to_device(batch, device)
                loss: torch.Tensor = student.compute_loss(batch)[0]
                loss.backward()

                # Clip to a big value if we don't want to clip the gradients
                # so that we can still log the gradient norms
                grad_norm = nn.utils.clip_grad_norm_(
                    student.parameters(),
                    1 if cfg.clip_grad_norm else 1000,
                ).item()
                grad_norms.append(grad_norm)

                # Step the optimizers
                optimizer_student.step()
                gradient_steps += 1
                training_samples += cfg.batch_size

                # Log losses
                loss_cpu = loss.item()
                demo_epoch_loss.append(loss_cpu)
                tepoch.set_postfix(base_loss=loss_cpu)

        mean_loss = np.mean(demo_epoch_loss)

        wandb.log(
            {
                "loss": mean_loss,
                "grad_norm": np.mean(grad_norms),
                "epoch": training_iterations,
                "iteration": iteration,
                "gradient_steps": gradient_steps,
                "training_samples": training_samples,
                "n_timesteps_in_buffer": buffer.size,
                "n_trajectories_in_buffer": buffer.n_trajectories,
                "beta": beta_to_use,
                "learning_rate_student": optimizer_student.param_groups[0]["lr"],
            },
            step=global_step,
        )

        # Step the learning rate scheduler
        lr_scheduler_student.step()
        student.eval()

        # Print some stats at the end of the iteration
        print(
            f"Iteration {iteration}/{cfg.num_iterations}, global step {global_step}, SPS {sps}"
        )

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()
