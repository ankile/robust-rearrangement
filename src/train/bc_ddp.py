import furniture_bench

from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from src.behavior.base import Actor
from src.common.context import suppress_stdout
from src.eval.eval_utils import get_model_from_api_or_cached
from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv


import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from src.dataset.dataset import (
    ImageDataset,
    StateDataset,
)
from src.eval.rollout import do_rollout_evaluation
from src.gym import get_env, get_rl_env
from tqdm import tqdm, trange
from ipdb import set_trace as bp
from src.behavior import get_actor
from src.dataset.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_to_device
from torch.utils.data import random_split, DataLoader
from src.common.earlystop import EarlyStopper
from src.common.files import get_processed_paths, path_override
from src.models.ema import SwitchEMA

from gym import logger

import hydra
from omegaconf import DictConfig, OmegaConf


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist


from wandb_osh.hooks import TriggerWandbSyncHook, _comm_default_dir

trigger_sync = TriggerWandbSyncHook(
    communication_dir=os.environ.get("WANDB_OSH_COMM_DIR", _comm_default_dir),
)

logger.set_level(logger.DISABLED)
OmegaConf.register_new_resolver("eval", eval)


print("=== Activate TF32 training? Deactivated for now...")
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


def ddp_setup():
    """
    Using torchrun so these things are set automatically
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def log_action_mse(log_dict, category, pred_action, gt_action):
    B, T, _ = pred_action.shape
    pred_action = pred_action.view(B, T, -1, 10)
    gt_action = gt_action.view(B, T, -1, 10)
    log_dict[f"action_sample/{category}_action_mse_error"] = (
        torch.nn.functional.mse_loss(pred_action, gt_action)
    )
    log_dict[f"action_sample/{category}_action_mse_error_pos"] = (
        torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
    )
    log_dict[f"action_sample/{category}_action_mse_error_rot"] = (
        torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
    )
    log_dict[f"action_sample/{category}_action_mse_error_width"] = (
        torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
    )


def to_native(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj


def set_dryrun_params(cfg: DictConfig):
    if cfg.dryrun:
        OmegaConf.set_struct(cfg, False)
        cfg.training.steps_per_epoch = 10 if cfg.training.steps_per_epoch != -1 else -1
        cfg.data.data_subset = 5
        cfg.data.dataloader_workers = 0
        cfg.training.sample_every = 1

        if cfg.rollout.rollouts:
            cfg.rollout.every = 1
            cfg.rollout.num_rollouts = 1
            cfg.rollout.loss_threshold = float("inf")
            cfg.rollout.max_steps = 10

        cfg.wandb.mode = "disabled"

        OmegaConf.set_struct(cfg, True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


@hydra.main(config_path="../config", config_name="base")
def main(cfg: DictConfig):
    set_dryrun_params(cfg)
    OmegaConf.resolve(cfg)
    env = None

    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", gpu_id)

    if gpu_id == 0:
        print(OmegaConf.to_yaml(cfg))

    if cfg.data.data_paths_override is None:
        data_path = get_processed_paths(
            controller=to_native(cfg.control.controller),
            domain=to_native(cfg.data.environment),
            task=to_native(cfg.data.furniture),
            demo_source=to_native(cfg.data.demo_source),
            randomness=to_native(cfg.data.randomness),
            demo_outcome=to_native(cfg.data.demo_outcome),
            suffix=to_native(cfg.data.suffix),
        )
    else:
        data_path = path_override(cfg.data.data_paths_override)

    print(f"Using data from {data_path}")

    if cfg.observation_type == "image":
        dataset = ImageDataset(
            dataset_paths=data_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            data_subset=cfg.data.data_subset,
            control_mode=cfg.control.control_mode,
            predict_past_actions=cfg.data.predict_past_actions,
            pad_after=cfg.data.get("pad_after", True),
            max_episode_count=cfg.data.get("max_episode_count", None),
            minority_class_power=cfg.data.get("minority_class_power", False),
            load_into_memory=cfg.data.get("load_into_memory", True),
        )
    elif cfg.observation_type == "state":
        dataset = StateDataset(
            dataset_paths=data_path,
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
    else:
        raise ValueError(f"Unknown observation type: {cfg.observation_type}")

    # Split the dataset into train and test (effective, meaning that this is after upsampling)
    train_size = int(len(dataset) * (1 - cfg.data.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    OmegaConf.set_struct(cfg, False)
    cfg.robot_state_dim = dataset.robot_state_dim

    if cfg.observation_type == "state":
        cfg.parts_poses_dim = dataset.parts_poses_dim

    # Create the policy network
    actor: Actor = get_actor(
        cfg,
        device,
    )
    actor.set_normalizer(dataset.normalizer)
    actor.to(device)

    actor: DDP = DDP(actor, device_ids=[gpu_id])

    # Set the data path in the cfg object
    cfg.data_path = [str(f) for f in data_path]

    # Update the cfg object with the action dimension
    cfg.action_dim = dataset.action_dim
    cfg.n_episodes = len(dataset.episode_ends)
    cfg.n_samples = dataset.n_samples

    # Update the cfg object with the observation dimension
    cfg.timestep_obs_dim = actor.module.timestep_obs_dim
    OmegaConf.set_struct(cfg, True)

    # Create dataloaders
    trainload_kwargs = dict(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        sampler=DistributedSampler(train_dataset),
    )
    trainloader = (
        FixedStepsDataloader(**trainload_kwargs, n_batches=cfg.training.steps_per_epoch)
        if cfg.training.steps_per_epoch != -1
        else DataLoader(**trainload_kwargs)
    )

    testload_kwargs = dict(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    testloader = (
        FixedStepsDataloader(
            **testload_kwargs,
            n_batches=max(
                int(round(cfg.training.steps_per_epoch * cfg.data.test_split)), 1
            ),
        )
        if cfg.training.steps_per_epoch != -1
        else DataLoader(**testload_kwargs)
    )

    # Create lists for optimizers and lr schedulers
    opt_noise = torch.optim.AdamW(
        params=actor.module.actor_parameters(),
        lr=cfg.training.actor_lr,
        weight_decay=cfg.regularization.weight_decay,
    )
    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=opt_noise,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * cfg.training.num_epochs,
    )

    optimizers = [("actor", opt_noise)]
    lr_schedulers = [lr_scheduler]

    if cfg.observation_type == "image":

        opt_encoder = torch.optim.AdamW(
            params=actor.module.encoder_parameters(),
            lr=cfg.training.encoder_lr,
            weight_decay=cfg.regularization.weight_decay,
        )
        lr_scheduler_encoder = get_scheduler(
            name=cfg.lr_scheduler.name,
            optimizer=opt_encoder,
            num_warmup_steps=cfg.lr_scheduler.encoder_warmup_steps,
            num_training_steps=len(trainloader) * cfg.training.num_epochs,
        )

        optimizers.append(("encoder", opt_encoder))
        lr_schedulers.append(lr_scheduler_encoder)

    if cfg.training.ema.use:
        ema = SwitchEMA(actor.module, cfg.training.ema.decay)
        ema.register()

    early_stopper = EarlyStopper(
        patience=cfg.early_stopper.patience,
        smooth_factor=cfg.early_stopper.smooth_factor,
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    starttime = now()
    print(f"Job started at: {starttime}")
    print(f"This process has access to {os.cpu_count()} CPUs.")

    # Init wandb
    if gpu_id == 0:
        run = wandb.init(
            id=cfg.wandb.continue_run_id,
            name=cfg.wandb.name,
            resume=None if cfg.wandb.continue_run_id is None else "must",
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity", "ankile"),
            config=config_dict,
            mode=cfg.wandb.mode,
            notes=cfg.wandb.notes,
        )

        # In sweeps, the init is ignored, so to make sure that the cfg is saved correctly
        # to wandb we need to log it manually
        wandb.config.update(config_dict)

        # save stats to wandb and update the cfg object
        train_size = int(dataset.n_samples * (1 - cfg.data.test_split))
        test_size = dataset.n_samples - train_size

        dataset_stats = {
            "num_samples_train": train_size,
            "num_samples_test": test_size,
            "num_episodes_train": int(
                len(dataset.episode_ends) * (1 - cfg.data.test_split)
            ),
            "num_episodes_test": int(len(dataset.episode_ends) * cfg.data.test_split),
            "dataset_metadata": dataset.metadata,
        }

        # Add the dataset stats to the wandb summary
        wandb.summary.update(dataset_stats)

        wandb.summary["start_time"] = starttime

        # Print the run name and storage location
        print(f"Run name: {run.name}")
        print(f"Run storage location: {run.dir}")

        # Create model save dir
        model_save_dir = Path(cfg.training.model_save_dir) / wandb.run.name
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Handle checkpoint loading and auto-resume properly here

    # Train loop
    best_test_loss = float("inf")
    test_loss_mean = float("inf")
    best_success_rate = 0
    prev_best_success_rate = 0
    global_step = 0

    early_stop = False

    # Wait here until all ranks are ready to begin training
    dist.barrier()

    pbar_desc = f"Epoch ({cfg.furniture}, {cfg.observation_type}{f', {cfg.vision_encoder.model}' if cfg.observation_type == 'image' else ''})"
    tglobal = trange(
        cfg.training.start_epoch,
        cfg.training.num_epochs,
        initial=cfg.training.start_epoch,
        total=cfg.training.num_epochs,
        desc=pbar_desc,
        position=0,
    )

    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        epoch_log = {
            "epoch": epoch_idx,
        }

        # batch loop
        actor.train()
        trainloader.sampler.set_epoch(epoch_idx)

        tepoch = tqdm(
            trainloader,
            desc=f"Training, GPU: {gpu_id}",
            position=gpu_id + 1,
            leave=False,
            total=len(trainloader),
        )
        for batch in tepoch:
            # Zero the gradients in all optimizers
            for _, opt in optimizers:
                opt.zero_grad()

            # Get a batch on device and compute loss and gradients
            batch = dict_to_device(batch, device)
            loss, losses_log = actor(batch)
            loss.backward()

            # Gradient clipping
            if cfg.training.clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=1.0
                )
            else:
                grad_norm = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2)
                            for p in actor.parameters()
                            if p.grad is not None
                        ]
                    ),
                    2,
                )

            # Step the optimizers and schedulers
            for (_, opt), scheduler in zip(optimizers, lr_schedulers):
                opt.step()
                scheduler.step()

            if cfg.training.ema.use:
                ema.update()

            # Log the loss and gradients
            loss_cpu = loss.item()
            step_log = {
                "batch_loss": loss_cpu,
                "grad_norm": grad_norm,
                **losses_log,
            }
            epoch_loss.append(loss_cpu)

            # Add the learning rates to the log
            for name, opt in optimizers:
                step_log[f"{name}_lr"] = opt.param_groups[0]["lr"]

            if gpu_id == 0:
                wandb.log(step_log, step=global_step)

            # Update the global step
            global_step += 1

            tepoch.set_postfix(loss=loss_cpu)

        tepoch.close()

        epoch_log["epoch_loss"] = np.mean(epoch_loss)

        # Only run logging and eval on the master process
        if (
            cfg.training.eval_every > 0
            and (epoch_idx + 1) % cfg.training.eval_every == 0
            and gpu_id == 0
        ):
            # Evaluation loop
            actor.eval()

            if cfg.training.ema.use:
                ema.apply_shadow()

            eval_losses_log = defaultdict(list)

            test_tepoch = tqdm(testloader, desc="Validation", leave=False)
            for test_batch in test_tepoch:
                with torch.no_grad():
                    # device transfer for test_batch
                    test_batch = dict_to_device(test_batch, device)

                    # Get test loss
                    test_loss_val, losses_log = actor.module.compute_loss(test_batch)

                    # logging
                    test_loss_cpu = test_loss_val.item()
                    test_loss.append(test_loss_cpu)
                    test_tepoch.set_postfix(loss=test_loss_cpu)

                    # Append the losses to the log
                    for k, v in losses_log.items():
                        eval_losses_log[k].append(v)

            test_tepoch.close()

            epoch_log["test_epoch_loss"] = test_loss_mean = np.mean(test_loss)
            # Update the epoch log with the mean of the evaluation losses

            for k, v in eval_losses_log.items():
                epoch_log[f"test_{k}"] = np.mean(v)

            if (
                cfg.rollout.rollouts
                and (epoch_idx + 1) % cfg.rollout.every == 0
                and np.mean(test_loss_mean) < cfg.rollout.loss_threshold
            ):
                # Do not load the environment until we successfuly made it this far
                if env is None:
                    env: FurnitureRLSimEnv = get_rl_env(
                        0,
                        furniture=cfg.rollout.furniture,
                        num_envs=cfg.rollout.num_envs,
                        randomness=cfg.rollout.randomness,
                        observation_space=cfg.observation_type,
                        resize_img=False,
                        act_rot_repr=cfg.control.act_rot_repr,
                        ctrl_mode=cfg.control.controller,
                        action_type=cfg.control.control_mode,
                        parts_poses_in_robot_frame=cfg.rollout.parts_poses_in_robot_frame,
                        headless=True,
                        verbose=True,
                    )

                best_success_rate = do_rollout_evaluation(
                    config=cfg,
                    env=env,
                    save_rollouts_to_file=cfg.rollout.save_rollouts,
                    save_rollouts_to_wandb=False,
                    actor=actor.module,
                    best_success_rate=best_success_rate,
                    epoch_idx=epoch_idx,
                )

            # Prepare the save dict once and we can reuse below
            actor_state = (
                ema.shadow if cfg.training.ema.use else actor.module.state_dict()
            )
            save_dict = {
                "model_state_dict": actor_state,
                "best_test_loss": best_test_loss,
                "best_success_rate": best_success_rate,
                "epoch": epoch_idx,
                "global_step": global_step,
            }

            # Save the model if the test loss is the best so far
            if (
                cfg.training.store_best_test_loss_model
                and test_loss_mean < best_test_loss
            ):
                best_test_loss = test_loss_mean
                save_path = str(model_save_dir / f"actor_chkpt_best_test_loss.pt")
                torch.save(save_dict, save_path)
                wandb.save(save_path)

            # Save the model if the success rate is the best so far
            if (
                cfg.training.store_best_success_rate_model
                and best_success_rate > prev_best_success_rate
            ):
                prev_best_success_rate = best_success_rate
                save_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(save_dict, save_path)
                wandb.save(save_path)

            if (
                cfg.training.checkpoint_interval > 0
                and (epoch_idx + 1) % cfg.training.checkpoint_interval == 0
            ):
                save_path = str(model_save_dir / f"actor_chkpt_{epoch_idx}.pt")
                torch.save(save_dict, save_path)
                wandb.save(save_path)

            if cfg.training.store_last_model:
                save_path = str(model_save_dir / f"actor_chkpt_last.pt")

                # Add the optimizer and scheduler states to the save dict
                # NOTE: We only do it here because this is the model we'll use for resuming runs
                # and as such will save disk space
                for (name, opt), scheduler in zip(optimizers, lr_schedulers):
                    save_dict[f"{name}_optimizer_state_dict"] = opt.state_dict()
                    save_dict[f"{name}_scheduler_state_dict"] = scheduler.state_dict()

                torch.save(save_dict, save_path)
                wandb.save(save_path)

            # Run diffusion sampling on a training batch
            if (
                cfg.training.sample_every > 0
                and (epoch_idx + 1) % cfg.training.sample_every == 0
            ):

                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    train_sampling_batch = dict_to_device(
                        next(iter(trainloader)), device
                    )
                    pred_action = actor.module.action_pred(train_sampling_batch)
                    gt_action = actor.module.normalizer(
                        train_sampling_batch["action"], "action", forward=False
                    )
                    log_action_mse(epoch_log, "train", pred_action, gt_action)

                    val_sampling_batch = dict_to_device(next(iter(testloader)), device)
                    gt_action = actor.module.normalizer(
                        val_sampling_batch["action"], "action", forward=False
                    )
                    pred_action = actor.module.action_pred(val_sampling_batch)
                    log_action_mse(epoch_log, "val", pred_action, gt_action)

            # If using EMA, restore the model
            if cfg.training.ema.use:
                ema.restore()

            # Since we now have a new test loss, we can update the early stopper
            early_stop = early_stopper.update(test_loss_mean)
            epoch_log["early_stopper/counter"] = early_stopper.counter
            epoch_log["early_stopper/best_loss"] = early_stopper.best_loss
            epoch_log["early_stopper/ema_loss"] = early_stopper.ema_loss

        # If switch is enabled, copy the the shadow to the model at the end of each epoch
        if cfg.training.ema.use and cfg.training.ema.switch:
            ema.copy_to_model()

        # Log epoch stats
        if gpu_id == 0:
            wandb.log(epoch_log, step=global_step)

        tglobal.set_postfix(
            time=now(),
            loss=epoch_log["epoch_loss"],
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
            stopper_counter=early_stopper.counter,
        )

        # If we are in offline mode, trigger the sync
        if (
            gpu_id == 0
            and cfg.wandb.mode == "offline"
            and (epoch_idx % cfg.wandb.get("osh_sync_interval", 1)) == 0
        ):
            trigger_sync()

        # Now that everything is logged and restored, we can check if we need to stop
        if early_stop:
            print(
                f"Early stopping at epoch {epoch_idx} as test loss did not improve for {early_stopper.patience} epochs."
            )
            break

        # Barrier to allow all processes to finish before next epoch
        dist.barrier()

    tglobal.close()
    wandb.finish()
    destroy_process_group()


if __name__ == "__main__":
    main()
