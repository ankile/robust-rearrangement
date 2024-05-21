from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from src.common.context import suppress_stdout
from src.gym.furniture_sim_env import FurnitureRLSimEnv

with suppress_stdout():
    import furniture_bench

import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from src.dataset.dataset import (
    FurnitureImageDataset,
    FurnitureStateDataset,
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
from src.common.files import get_processed_paths
from src.models.ema import SwitchEMA

from gym import logger

import hydra
from omegaconf import DictConfig, OmegaConf

from wandb_osh.hooks import TriggerWandbSyncHook, _comm_default_dir

trigger_sync = TriggerWandbSyncHook(
    communication_dir=os.environ.get("WANDB_OSH_COMM_DIR", _comm_default_dir),
)

logger.set_level(logger.DISABLED)
OmegaConf.register_new_resolver("eval", eval)


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


@hydra.main(config_path="../config/bc", config_name="base")
def main(cfg: DictConfig):
    set_dryrun_params(cfg)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    env = None
    device = torch.device(
        f"cuda:{cfg.training.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    data_path = get_processed_paths(
        controller=to_native(cfg.control.controller),
        domain=to_native(cfg.data.environment),
        task=to_native(cfg.data.furniture),
        demo_source=to_native(cfg.data.demo_source),
        randomness=to_native(cfg.data.randomness),
        demo_outcome=to_native(cfg.data.demo_outcome),
        suffix=to_native(cfg.data.suffix),
    )

    print(f"Using data from {data_path}")

    if cfg.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_paths=data_path,
            pred_horizon=cfg.data.pred_horizon,
            obs_horizon=cfg.data.obs_horizon,
            action_horizon=cfg.data.action_horizon,
            data_subset=cfg.data.data_subset,
            control_mode=cfg.control.control_mode,
            predict_past_actions=cfg.data.predict_past_actions,
            pad_after=cfg.data.get("pad_after", True),
            max_episode_count=cfg.data.get("max_episode_count", None),
        )
    elif cfg.observation_type == "state":
        dataset = FurnitureStateDataset(
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

    # Split the dataset into train and test
    train_size = int(len(dataset) * (1 - cfg.data.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    OmegaConf.set_struct(cfg, False)
    cfg.robot_state_dim = dataset.robot_state_dim

    if cfg.observation_type == "state":
        cfg.parts_poses_dim = dataset.parts_poses_dim

    # Create the policy network
    actor = get_actor(
        cfg,
        device,
    )
    actor.set_normalizer(dataset.normalizer)
    actor.to(device)

    # Set the data path in the cfg object
    cfg.data_path = [str(f) for f in data_path]

    # Update the cfg object with the action dimension
    cfg.action_dim = dataset.action_dim
    cfg.n_episodes = len(dataset.episode_ends)
    cfg.n_samples = len(dataset)

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

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    # AdamW optimizer for noise_net
    opt_noise = torch.optim.AdamW(
        params=actor.parameters(),
        lr=cfg.training.actor_lr,
        weight_decay=cfg.regularization.weight_decay,
    )

    if cfg.training.ema.use:
        ema = SwitchEMA(actor, cfg.training.ema.decay)
        ema.register()

    n_batches = len(trainloader)

    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=opt_noise,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * cfg.training.num_epochs,
    )

    early_stopper = EarlyStopper(
        patience=cfg.early_stopper.patience,
        smooth_factor=cfg.early_stopper.smooth_factor,
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # Init wandb
    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        name=cfg.wandb.name,
        resume=cfg.wandb.continue_run_id is not None,
        project=cfg.wandb.project,
        entity="ankile",
        config=config_dict,
        mode=cfg.wandb.mode,
        notes=cfg.wandb.notes,
    )

    # Print the run name and storage location
    print(f"Run name: {run.name}")
    print(f"Run storage location: {run.dir}")

    # In sweeps, the init is ignored, so to make sure that the cfg is saved correctly
    # to wandb we need to log it manually
    wandb.config.update(config_dict)

    # save stats to wandb and update the cfg object
    wandb.log(
        {
            "dataset/num_samples_train": len(train_dataset),
            "dataset/num_samples_test": len(test_dataset),
            "dataset/num_episodes_train": int(
                len(dataset.episode_ends) * (1 - cfg.data.test_split)
            ),
            "dataset/num_episodes_test": int(
                len(dataset.episode_ends) * cfg.data.test_split
            ),
            "dataset/dataset_metadata": dataset.metadata,
        }
    )

    # Create model save dir
    model_save_dir = Path(cfg.training.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    best_test_loss = float("inf")
    test_loss_mean = float("inf")
    best_success_rate = 0
    prev_best_success_rate = 0

    print(f"Job started at: {now()}")

    pbar_desc = f"Epoch ({cfg.furniture}, {cfg.observation_type}{f', {cfg.vision_encoder.model}' if cfg.observation_type == 'image' else ''})"

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

        # batch loop
        actor.train()
        tepoch = tqdm(trainloader, desc="Training", leave=False, total=n_batches)
        for batch in tepoch:
            opt_noise.zero_grad()

            # device transfer
            batch = dict_to_device(batch, device)

            # Get loss
            loss, losses_log = actor.compute_loss(batch)

            # backward pass
            loss.backward()

            # optimizer step
            opt_noise.step()
            lr_scheduler.step()

            if cfg.training.ema.use:
                ema.update()

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            lr = lr_scheduler.get_last_lr()[0]
            wandb.log(
                {
                    "training/lr": lr,
                    "batch_loss": loss_cpu,
                    **losses_log,
                }
            )

            tepoch.set_postfix(loss=loss_cpu, lr=lr)

        tepoch.close()

        epoch_log["epoch_loss"] = np.mean(epoch_loss)

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

        # Save the model if the test loss is the best so far
        if cfg.training.checkpoint_model and test_loss_mean < best_test_loss:
            best_test_loss = test_loss_mean
            save_path = str(model_save_dir / f"actor_chkpt_best_test_loss.pt")
            torch.save(
                actor.state_dict(),
                save_path,
            )
            wandb.save(save_path)

        if (
            cfg.training.checkpoint_model
            and (epoch_idx + 1) % cfg.training.checkpoint_interval == 0
        ):
            save_path = str(model_save_dir / f"actor_chkpt_{epoch_idx}.pt")
            torch.save(
                actor.state_dict(),
                save_path,
            )
            wandb.save(save_path)

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

        # run diffusion sampling on a training batch
        if ((epoch_idx + 1) % cfg.training.sample_every) == 0:
            with torch.no_grad():
                # sample trajectory from training set, and evaluate difference
                train_sampling_batch = dict_to_device(next(iter(trainloader)), device)
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

        if (
            cfg.rollout.rollouts
            and (epoch_idx + 1) % cfg.rollout.every == 0
            and np.mean(test_loss_mean) < cfg.rollout.loss_threshold
        ):
            # Do not load the environment until we successfuly made it this far
            if env is None:
                env: FurnitureRLSimEnv = get_rl_env(
                    cfg.training.gpu_id,
                    furniture=cfg.rollout.furniture,
                    num_envs=cfg.rollout.num_envs,
                    randomness=cfg.rollout.randomness,
                    observation_space=cfg.observation_type,
                    resize_img=False,
                    act_rot_repr=cfg.control.act_rot_repr,
                    ctrl_mode=cfg.control.controller,
                    action_type=cfg.control.control_mode,
                    headless=True,
                    # pos_scalar=1,
                    # rot_scalar=1,
                    # stiffness=1_000,
                    # damping=200,
                )

            best_success_rate = do_rollout_evaluation(
                cfg,
                env,
                cfg.rollout.save_rollouts,
                actor,
                best_success_rate,
                epoch_idx,
            )

            # Save the model if the success rate is the best so far
            if (
                cfg.training.checkpoint_model
                and best_success_rate > prev_best_success_rate
            ):
                prev_best_success_rate = best_success_rate
                save_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(
                    actor.state_dict(),
                    save_path,
                )
                wandb.save(save_path)

        # After eval is done, we restore the model to the original state
        if cfg.training.ema.use:
            # If using normal EMA, restore the model
            ema.restore()

            # If using switch EMA, set the model to the shadow
            if cfg.training.ema.switch:
                ema.copy_to_model()

        # Update the early stopper
        early_stop = early_stopper.update(test_loss_mean)
        epoch_log["early_stopper/counter"] = early_stopper.counter
        epoch_log["early_stopper/best_loss"] = early_stopper.best_loss
        epoch_log["early_stopper/ema_loss"] = early_stopper.ema_loss

        # Update the epoch log with the mean of the evaluation losses
        for k, v in eval_losses_log.items():
            epoch_log[f"test_{k}"] = np.mean(v)

        # Log epoch stats
        wandb.log(epoch_log)
        tglobal.set_postfix(
            time=now(),
            loss=epoch_log["epoch_loss"],
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
            stopper_counter=early_stopper.counter,
        )

        # If we are in offline mode, trigger the sync
        if cfg.wandb.mode == "offline":
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
