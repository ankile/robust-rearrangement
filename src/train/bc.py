import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import hydra
import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from gymnasium import Env
from ipdb import set_trace as bp
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange

from src.behavior import get_actor
from src.behavior.base import Actor
from src.common.earlystop import EarlyStopper
from src.common.files import get_processed_paths, path_override
from src.common.hydra import to_native
from src.common.pytorch_util import dict_to_device
from src.dataset.dataloader import FixedStepsDataloader
from src.dataset.dataset import ImageDataset, StateDataset
from src.eval.eval_utils import get_model_from_api_or_cached
from src.eval.rollout import do_rollout_evaluation
from src.gym import get_rl_env
from src.models.ema import SwitchEMA

# Import the wandb Run type for type hinting
from wandb.apis.public.runs import Run
from wandb.errors.util import CommError
from wandb_osh.hooks import TriggerWandbSyncHook, _comm_default_dir

trigger_sync = TriggerWandbSyncHook(
    communication_dir=os.environ.get("WANDB_OSH_COMM_DIR", _comm_default_dir),
)


print("=== Activate TF32 training? Deactivated for now...")
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


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


def set_dryrun_params(cfg: DictConfig):
    if cfg.dryrun:
        OmegaConf.set_struct(cfg, False)
        cfg.training.steps_per_epoch = 10 if cfg.training.steps_per_epoch != -1 else -1
        cfg.data.data_subset = 5
        cfg.data.dataloader_workers = 0
        cfg.training.sample_every = 1
        cfg.training.eval_every = 1

        if cfg.rollout.rollouts:
            cfg.rollout.every = 1
            # cfg.rollout.num_rollouts = 1
            cfg.rollout.loss_threshold = float("inf")
            # cfg.rollout.max_steps = 10

        cfg.wandb.mode = "disabled"

        OmegaConf.set_struct(cfg, True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# @hydra.main(config_path="../config/bc", config_name="base")
@hydra.main(config_path="../config", config_name="base")
def main(cfg: DictConfig):
    set_dryrun_params(cfg)
    OmegaConf.resolve(cfg)

    # Set the random seed
    if cfg.get("seed") is None:
        OmegaConf.set_struct(cfg, False)
        cfg.seed = np.random.randint(0, 2**32 - 1)
        OmegaConf.set_struct(cfg, True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))
    env: Optional[Env] = None
    device = torch.device(
        f"cuda:{cfg.training.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    state_dict = None

    # Check if we are continuing a run
    run_exists = False
    if cfg.wandb.continue_run_id is not None:
        try:
            run: Run = wandb.Api().run(
                f"{cfg.wandb.project}/{cfg.wandb.continue_run_id}"
            )
            run_exists = True
        except (ValueError, CommError):
            pass

    if run_exists:
        print(f"Continuing run {cfg.wandb.continue_run_id}, {run.name}")

        run_id = cfg.wandb.continue_run_id
        run_path = f"{cfg.wandb.project}/{run_id}"
        wandb_mode = cfg.wandb.mode

        data_paths_override = cfg.data.data_paths_override

        # Load the weights from the run and override the config with the one from the run
        try:
            cfg, wts = get_model_from_api_or_cached(
                run_path, "last", wandb_mode=wandb_mode
            )
        except:
            cfg, wts = get_model_from_api_or_cached(
                run_path, "latest", wandb_mode=wandb_mode
            )

        # Ensure we set the `continue_run_id` to the run_id
        cfg.wandb.continue_run_id = run_id
        cfg.wandb.mode = wandb_mode
        cfg.data.data_paths_override = data_paths_override

        state_dict = torch.load(wts)

        epoch_idx = state_dict.get("epoch", run.summary.get("epoch", 0))
        cfg.training.start_epoch = epoch_idx

        # Set the best test loss and success rate to the one from the run
        best_test_loss = state_dict.get(
            "best_test_loss", run.summary.get("test_epoch_loss", float("inf"))
        )
        test_loss_mean = best_test_loss
        best_success_rate = state_dict.get(
            "best_success_rate", run.summary.get("best_success_rate", 0)
        )
        epoch_idx = state_dict.get("epoch", run.summary.get("epoch", 0))
        global_step = state_dict.get("global_step", run.lastHistoryStep)

        prev_best_success_rate = best_success_rate
    else:
        # Train loop
        best_test_loss = float("inf")
        test_loss_mean = float("inf")
        best_success_rate = 0
        prev_best_success_rate = 0
        global_step = 0

    if cfg.data.data_paths_override is None:
        data_path = get_processed_paths(
            controller=to_native(cfg.control.controller),
            domain=to_native(cfg.data.environment),
            task=to_native(cfg.data.task),
            demo_source=to_native(cfg.data.demo_source),
            randomness=to_native(cfg.data.randomness),
            demo_outcome=to_native(cfg.data.demo_outcome),
            suffix=to_native(cfg.data.suffix),
        )
    else:
        data_path = path_override(cfg.data.data_paths_override)

    print(f"Using data from {data_path}")

    dataset: Union[ImageDataset, StateDataset]

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
    if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
        cfg.slurm_job_id = job_id

    cfg.robot_state_dim = dataset.robot_state_dim
    cfg.action_dim = dataset.action_dim

    if cfg.observation_type == "state":
        cfg.parts_poses_dim = dataset.parts_poses_dim

    # Create the policy network
    actor: Actor = get_actor(
        cfg,
        device,
    )
    actor.set_normalizer(dataset.normalizer)
    actor.to(device)

    # Set the data path in the cfg object
    cfg.data_path = [str(f) for f in data_path]

    # Update the cfg object with the action dimension
    cfg.n_episodes = len(dataset.episode_ends)
    cfg.n_samples = dataset.n_samples

    # Update the cfg object with the observation dimension
    cfg.timestep_obs_dim = actor.timestep_obs_dim
    OmegaConf.set_struct(cfg, True)

    if cfg.training.load_checkpoint_run_id is not None:
        api = wandb.Api()
        run = api.run(cfg.training.load_checkpoint_run_id)
        model_path = (
            [f for f in run.files() if f.name.endswith(".pt")][0]
            .download(exist_ok=True)
            .name
        )
        print(f"Loading checkpoint from {cfg.training.load_checkpoint_run_id}")
        actor.load_state_dict(torch.load(model_path))

    # Create dataloaders
    trainload_kwargs = dict(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
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
        params=actor.actor_parameters(),
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
            params=actor.encoder_parameters(),
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

    if state_dict is not None:
        if "model_state_dict" in state_dict:
            actor.load_state_dict(state_dict["model_state_dict"])
            for (name, opt), scheduler in zip(optimizers, lr_schedulers):
                opt.load_state_dict(state_dict[f"{name}_optimizer_state_dict"])
                scheduler.load_state_dict(state_dict[f"{name}_scheduler_state_dict"])

        else:
            actor.load_state_dict(state_dict)

        print(f"Loaded weights from run {run_id}")

    if cfg.training.ema.use:
        ema = SwitchEMA(actor, cfg.training.ema.decay)
        ema.register()

    early_stopper = EarlyStopper(
        patience=cfg.early_stopper.patience,
        smooth_factor=cfg.early_stopper.smooth_factor,
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Init wandb
    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        name=cfg.wandb.name,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        config=config_dict,
        mode=cfg.wandb.mode,
        notes=cfg.wandb.notes,
    )

    if cfg.wandb.watch_model:
        run.watch(actor, log="all", log_freq=1000)

    # Print the run name and storage location
    print(f"Run name: {run.name}")
    print(f"Run storage location: {run.dir}")

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

    starttime = now()
    wandb.summary["start_time"] = starttime

    # Create model save dir
    model_save_dir = Path(cfg.training.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Job started at: {starttime}")

    early_stop = False

    pbar_desc = f"Epoch ({cfg.task}, {cfg.observation_type}{f', {cfg.vision_encoder.model}' if cfg.observation_type == 'image' else ''})"

    tglobal = trange(
        cfg.training.start_epoch,
        cfg.training.num_epochs,
        initial=cfg.training.start_epoch,
        total=cfg.training.num_epochs,
        desc=pbar_desc,
    )

    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        epoch_log = {
            "epoch": epoch_idx,
        }

        train_losses_log = defaultdict(list)

        # batch loop
        actor.train()
        tepoch = tqdm(trainloader, desc="Training", leave=False, total=len(trainloader))
        for batch in tepoch:
            # Zero the gradients in all optimizers
            for _, opt in optimizers:
                opt.zero_grad()

            # Get a batch on device and compute loss and gradients
            batch = dict_to_device(batch, device)
            loss, losses_log = actor.compute_loss(batch)
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=1.0 + 1e3 * (1 - cfg.training.clip_grad_norm),
            )

            # Step the optimizers and schedulers
            for (_, opt), scheduler in zip(optimizers, lr_schedulers):
                opt.step()
                scheduler.step()

            if cfg.training.ema.use:
                ema.update()

            # Log the loss and gradients
            loss_cpu = loss.item()

            train_losses_log["grad_norm"] = grad_norm.item()

            for k, v in losses_log.items():
                train_losses_log[k].append(v)

            epoch_loss.append(loss_cpu)

            # Update the global step
            global_step += 1

            tepoch.set_postfix(loss=loss_cpu)

        tepoch.close()

        epoch_log["epoch_loss"] = np.mean(epoch_loss)

        for k, v in train_losses_log.items():
            epoch_log[f"train_{k}"] = np.mean(v)

        # Add the learning rates to the log
        for name, opt in optimizers:
            epoch_log[f"{name}_lr"] = opt.param_groups[0]["lr"]

        # Prepare the save dict once and we can reuse below
        save_dict = {
            "model_state_dict": actor.state_dict(),
            "best_test_loss": best_test_loss,
            "best_success_rate": best_success_rate,
            "epoch": epoch_idx,
            "global_step": global_step,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        # Add the optimizer and scheduler states to the save dict
        for (name, opt), scheduler in zip(optimizers, lr_schedulers):
            save_dict[f"{name}_optimizer_state_dict"] = opt.state_dict()
            save_dict[f"{name}_scheduler_state_dict"] = scheduler.state_dict()

        if (
            cfg.training.eval_every > 0
            and (epoch_idx + 1) % cfg.training.eval_every == 0
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
                    test_loss_val, losses_log = actor.compute_loss(test_batch)

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
                    env = get_rl_env(
                        cfg.training.gpu_id,
                        task=cfg.rollout.task,
                        num_envs=cfg.rollout.num_envs,
                        randomness=cfg.rollout.randomness,
                        observation_space=cfg.observation_type,
                        resize_img=False,
                        act_rot_repr=cfg.control.act_rot_repr,
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
                    actor=actor,
                    best_success_rate=best_success_rate,
                    epoch_idx=epoch_idx,
                )

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
                    pred_action = actor.action_pred(train_sampling_batch)
                    gt_action = actor.normalizer(
                        train_sampling_batch["action"], "action", forward=False
                    )
                    log_action_mse(epoch_log, "train", pred_action, gt_action)

                    val_sampling_batch = dict_to_device(next(iter(testloader)), device)
                    gt_action = actor.normalizer(
                        val_sampling_batch["action"], "action", forward=False
                    )
                    pred_action = actor.action_pred(val_sampling_batch)
                    log_action_mse(epoch_log, "val", pred_action, gt_action)

            # If using EMA, restore the model
            if cfg.training.ema.use:
                ema.restore()

            # Since we now have a new test loss, we can update the early stopper
            early_stop = early_stopper.update(test_loss_mean)
            epoch_log["early_stopper/counter"] = early_stopper.counter
            epoch_log["early_stopper/best_loss"] = early_stopper.best_loss
            epoch_log["early_stopper/ema_loss"] = early_stopper.ema_loss

        # We store the last model at the end of each epoch for better checkpointing
        if cfg.training.store_last_model:
            save_path = str(model_save_dir / f"actor_chkpt_last.pt")
            torch.save(save_dict, save_path)
            wandb.save(save_path)

        # If switch is enabled, copy the the shadow to the model at the end of each epoch
        if cfg.training.ema.use and cfg.training.ema.switch:
            ema.copy_to_model()

        # Log epoch stats
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
            cfg.wandb.mode == "offline"
            and (epoch_idx % cfg.wandb.get("osh_sync_interval", 1)) == 0
        ):
            trigger_sync()

        # Now that everything is logged and restored, we can check if we need to stop
        if early_stop:
            print(
                f"Early stopping at epoch {epoch_idx} as test loss did not improve for {early_stopper.patience} epochs."
            )
            break

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    main()
