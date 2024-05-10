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


def set_dryrun_params(config: DictConfig):
    if config.dryrun:
        OmegaConf.set_struct(config, False)
        config.training.steps_per_epoch = (
            10 if config.training.steps_per_epoch != -1 else -1
        )
        config.data.data_subset = 5
        config.data.dataloader_workers = 0

        if config.rollout.rollouts:
            config.rollout.every = 1
            config.rollout.num_rollouts = 1
            config.rollout.loss_threshold = float("inf")
            config.rollout.max_steps = 10

        config.wandb.mode = "disabled"

        OmegaConf.set_struct(config, True)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


@hydra.main(config_path="../config", config_name="base")
def main(config: DictConfig):
    set_dryrun_params(config)
    OmegaConf.resolve(config)
    print(OmegaConf.to_yaml(config))
    env = None
    device = torch.device(
        f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    data_path = get_processed_paths(
        controller=to_native(config.control.controller),
        domain=to_native(config.data.environment),
        task=to_native(config.data.furniture),
        demo_source=to_native(config.data.demo_source),
        randomness=to_native(config.data.randomness),
        demo_outcome=to_native(config.data.demo_outcome),
        suffix=to_native(config.data.suffix),
    )

    print(f"Using data from {data_path}")

    if config.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_paths=data_path,
            pred_horizon=config.data.pred_horizon,
            obs_horizon=config.data.obs_horizon,
            action_horizon=config.data.action_horizon,
            data_subset=config.data.data_subset,
            control_mode=config.control.control_mode,
            predict_past_actions=config.data.predict_past_actions,
            pad_after=config.data.get("pad_after", True),
            max_episode_count=config.data.get("max_episode_count", None),
        )
    elif config.observation_type == "state":
        dataset = FurnitureStateDataset(
            dataset_paths=data_path,
            pred_horizon=config.data.pred_horizon,
            obs_horizon=config.data.obs_horizon,
            action_horizon=config.data.action_horizon,
            data_subset=config.data.data_subset,
            control_mode=config.control.control_mode,
            predict_past_actions=config.data.predict_past_actions,
            pad_after=config.data.get("pad_after", True),
            max_episode_count=config.data.get("max_episode_count", None),
        )
    else:
        raise ValueError(f"Unknown observation type: {config.observation_type}")

    # Split the dataset into train and test
    train_size = int(len(dataset) * (1 - config.data.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    OmegaConf.set_struct(config, False)
    config.robot_state_dim = dataset.robot_state_dim

    if config.observation_type == "state":
        config.parts_poses_dim = dataset.parts_poses_dim

    # Create the policy network
    actor = get_actor(
        config,
        device,
    )
    actor.set_normalizer(dataset.normalizer)
    actor.to(device)

    # Set the data path in the config object
    config.data_path = [str(f) for f in data_path]

    # Update the config object with the action dimension
    config.action_dim = dataset.action_dim
    config.n_episodes = len(dataset.episode_ends)
    config.n_samples = len(dataset)

    # Update the config object with the observation dimension
    config.timestep_obs_dim = actor.timestep_obs_dim
    OmegaConf.set_struct(config, True)

    if config.training.load_checkpoint_run_id is not None:
        api = wandb.Api()
        run = api.run(config.training.load_checkpoint_run_id)
        model_path = (
            [f for f in run.files() if f.name.endswith(".pt")][0]
            .download(exist_ok=True)
            .name
        )
        print(f"Loading checkpoint from {config.training.load_checkpoint_run_id}")
        actor.load_state_dict(torch.load(model_path))

    # Create dataloaders
    trainload_kwargs = dict(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )
    trainloader = (
        FixedStepsDataloader(
            **trainload_kwargs, n_batches=config.training.steps_per_epoch
        )
        if config.training.steps_per_epoch != -1
        else DataLoader(**trainload_kwargs)
    )

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    # AdamW optimizer for noise_net
    opt_noise = torch.optim.AdamW(
        params=actor.parameters(),
        lr=config.training.actor_lr,
        weight_decay=config.regularization.weight_decay,
    )

    if config.training.ema.use:
        ema = SwitchEMA(actor, config.training.ema.decay)
        ema.register()

    n_batches = len(trainloader)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.name,
        optimizer=opt_noise,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * config.training.num_epochs,
    )

    early_stopper = EarlyStopper(
        patience=config.early_stopper.patience,
        smooth_factor=config.early_stopper.smooth_factor,
    )
    config_dict = OmegaConf.to_container(config, resolve=True)
    # Init wandb
    run = wandb.init(
        id=config.wandb.continue_run_id,
        name=config.wandb.name,
        resume=config.wandb.continue_run_id is not None,
        project=config.wandb.project,
        entity="ankile",
        config=config_dict,
        mode=config.wandb.mode,
        notes=config.wandb.notes,
    )

    # Print the run name and storage location
    print(f"Run name: {run.name}")
    print(f"Run storage location: {run.dir}")

    # In sweeps, the init is ignored, so to make sure that the config is saved correctly
    # to wandb we need to log it manually
    wandb.config.update(config_dict)

    # save stats to wandb and update the config object
    wandb.log(
        {
            "dataset/num_samples_train": len(train_dataset),
            "dataset/num_samples_test": len(test_dataset),
            "dataset/num_episodes_train": int(
                len(dataset.episode_ends) * (1 - config.data.test_split)
            ),
            "dataset/num_episodes_test": int(
                len(dataset.episode_ends) * config.data.test_split
            ),
            "dataset/dataset_metadata": dataset.metadata,
        }
    )

    # Create model save dir
    model_save_dir = Path(config.training.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    best_test_loss = float("inf")
    test_loss_mean = float("inf")
    best_success_rate = 0
    prev_best_success_rate = 0

    print(f"Job started at: {now()}")

    pbar_desc = f"Epoch ({config.furniture}, {config.observation_type}{f', {config.vision_encoder.model}' if config.observation_type == 'image' else ''})"

    tglobal = trange(
        config.training.start_epoch,
        config.training.num_epochs,
        initial=config.training.start_epoch,
        total=config.training.num_epochs,
        desc=pbar_desc,
    )
    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        # batch loop
        actor.train_mode()
        tepoch = tqdm(trainloader, desc="Training", leave=False, total=n_batches)
        for batch in tepoch:
            opt_noise.zero_grad()

            # device transfer
            # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            batch = dict_to_device(batch, device)

            # Get loss
            loss = actor.compute_loss(batch)

            # backward pass
            loss.backward()

            # optimizer step
            opt_noise.step()
            lr_scheduler.step()

            if config.training.ema.use:
                ema.update()

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            lr = lr_scheduler.get_last_lr()[0]
            wandb.log(
                {
                    "training/lr": lr,
                    "batch_loss": loss_cpu,
                }
            )

            tepoch.set_postfix(loss=loss_cpu, lr=lr)

        tepoch.close()

        train_loss_mean = np.mean(epoch_loss)

        # Evaluation loop
        actor.eval_mode()

        if config.training.ema.use:
            ema.apply_shadow()

        test_tepoch = tqdm(testloader, desc="Validation", leave=False)
        for test_batch in test_tepoch:
            with torch.no_grad():
                # device transfer for test_batch
                test_batch = dict_to_device(test_batch, device)

                # Get test loss
                test_loss_val = actor.compute_loss(test_batch)

                # logging
                test_loss_cpu = test_loss_val.item()
                test_loss.append(test_loss_cpu)
                test_tepoch.set_postfix(loss=test_loss_cpu)

        test_tepoch.close()

        test_loss_mean = np.mean(test_loss)

        # Save the model if the test loss is the best so far
        if config.training.checkpoint_model and test_loss_mean < best_test_loss:
            best_test_loss = test_loss_mean
            save_path = str(model_save_dir / f"actor_chkpt_best_test_loss.pt")
            torch.save(
                actor.state_dict(),
                save_path,
            )
            wandb.save(save_path)

        if (
            config.training.checkpoint_model
            and (epoch_idx + 1) % config.training.checkpoint_interval == 0
        ):
            save_path = str(model_save_dir / f"actor_chkpt_{epoch_idx}.pt")
            torch.save(
                actor.state_dict(),
                save_path,
            )
            wandb.save(save_path)

        # Early stopping
        if early_stopper.update(test_loss_mean):
            print(
                f"Early stopping at epoch {epoch_idx} as test loss did not improve for {early_stopper.patience} epochs."
            )
            # Log to wandb the final counter
            wandb.log(
                {
                    "early_stopper/counter": early_stopper.counter,
                    "early_stopper/best_loss": early_stopper.best_loss,
                    "early_stopper/ema_loss": early_stopper.ema_loss,
                }
            )
            break

        # Log epoch stats
        wandb.log(
            {
                "epoch": epoch_idx,
                "epoch_loss": np.mean(epoch_loss),
                "test_epoch_loss": test_loss_mean,
                "early_stopper/counter": early_stopper.counter,
                "early_stopper/best_loss": early_stopper.best_loss,
                "early_stopper/ema_loss": early_stopper.ema_loss,
            }
        )
        tglobal.set_postfix(
            time=now(),
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
            stopper_counter=early_stopper.counter,
        )

        if (
            config.rollout.rollouts
            and (epoch_idx + 1) % config.rollout.every == 0
            and np.mean(test_loss_mean) < config.rollout.loss_threshold
        ):
            # Do not load the environment until we successfuly made it this far
            if env is None:
                from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

                # env: FurnitureSimEnv = get_env(
                env: FurnitureRLSimEnv = get_rl_env(
                    config.training.gpu_id,
                    furniture=config.rollout.furniture,
                    num_envs=config.rollout.num_envs,
                    randomness=config.rollout.randomness,
                    observation_space=config.observation_type,
                    # Now using full size images in sim and resizing to be consistent
                    # observation_space=config.observation_type,
                    resize_img=False,
                    act_rot_repr=config.control.act_rot_repr,
                    ctrl_mode=config.control.controller,
                    action_type=config.control.control_mode,
                    headless=True,
                    pos_scalar=1,
                    rot_scalar=1,
                    stiffness=1_000,
                    damping=200,
                )

            best_success_rate = do_rollout_evaluation(
                config,
                env,
                config.rollout.save_rollouts,
                actor,
                best_success_rate,
                epoch_idx,
            )

            # Save the model if the success rate is the best so far
            if (
                config.training.checkpoint_model
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
            if config.training.ema.use:
                # If using normal EMA, restore the model
                ema.restore()

                # If using switch EMA, set the model to the shadow
                if config.training.ema.switch:
                    ema.copy_to_model()

        if config.wandb.mode == "offline":
            trigger_sync()

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    main()
