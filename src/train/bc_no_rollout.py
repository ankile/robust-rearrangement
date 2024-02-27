from math import e
from pathlib import Path

import numpy as np
import torch
import wandb

# from wandb_osh.hooks import TriggerWandbSyncHook

from diffusers.optimization import get_scheduler
from src.dataset.dataset import (
    FurnitureImageDataset,
    FurnitureFeatureDataset,
)
from src.dataset import get_normalizer
from tqdm import tqdm, trange
from ipdb import set_trace as bp
from src.behavior import get_actor
from src.dataset.dataloader import FixedStepsDataloader
from src.dataset.normalizer import Normalizer
from src.common.pytorch_util import dict_to_device
from torch.utils.data import random_split, DataLoader
from src.common.earlystop import EarlyStopper
from src.common.files import get_processed_paths

from datetime import datetime

from gym import logger

import hydra
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("eval", eval)

from wandb_osh.hooks import TriggerWandbSyncHook

trigger_sync = TriggerWandbSyncHook()

logger.set_level(logger.DISABLED)


def to_native(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj


def set_dryrun_params(config: DictConfig):
    if config.dryrun:
        OmegaConf.set_struct(config, False)
        config.training.steps_per_epoch = 10
        config.data.data_subset = 1

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
        environment=to_native(config.data.environment),
        task=to_native(config.data.furniture),
        demo_source=to_native(config.data.demo_source),
        randomness=to_native(config.data.randomness),
        demo_outcome=to_native(config.data.demo_outcome),
    )

    normalizer: Normalizer = get_normalizer(
        config.data.normalization, config.control.control_mode
    )

    print(f"Using data from {data_path}")

    if config.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_paths=data_path,
            pred_horizon=config.data.pred_horizon,
            obs_horizon=config.data.obs_horizon,
            action_horizon=config.data.action_horizon,
            normalizer=normalizer.get_copy(),
            augment_image=config.data.augment_image,
            data_subset=config.data.data_subset,
            control_mode=config.control.control_mode,
            first_action_idx=config.actor.first_action_index,
            pad_after=config.data.get("pad_after", True),
        )
    elif config.observation_type == "feature":
        raise ValueError("Feature observation type is not supported")
        dataset = FurnitureFeatureDataset(
            dataset_paths=data_path,
            pred_horizon=config.data.pred_horizon,
            obs_horizon=config.data.obs_horizon,
            action_horizon=config.data.action_horizon,
            normalizer=normalizer.get_copy(),
            encoder_name=config.vision_encoder.model,
            data_subset=config.data.data_subset,
            control_mode=config.control.control_mode,
            first_action_idx=config.actor.first_action_index,
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

    # Create the policy network
    actor = get_actor(
        config,
        normalizer.get_copy(),
        device,
    )

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
    trainloader = FixedStepsDataloader(
        dataset=train_dataset,
        n_batches=config.training.steps_per_epoch,
        batch_size=config.training.batch_size,
        num_workers=config.data.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
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
    wandb.init(
        id=config.wandb.continue_run_id,
        name=config.wandb.name,
        resume=config.wandb.continue_run_id is not None,
        project=config.wandb.project,
        entity="robot-rearrangement",
        config=config_dict,
        mode=config.wandb.mode,
        notes=config.wandb.notes,
    )

    # In sweeps, the init is ignored, so to make sure that the config is saved correctly
    # to wandb we need to log it manually
    wandb.config.update(config_dict)

    # save stats to wandb and update the config object
    wandb.log(
        {
            "num_samples_train": len(train_dataset),
            "num_samples_test": len(test_dataset),
            "num_episodes_train": int(
                len(dataset.episode_ends) * (1 - config.data.test_split)
            ),
            "num_episodes_test": int(
                len(dataset.episode_ends) * config.data.test_split
            ),
            "dataset_metadata": dataset.metadata,
        }
    )

    # Create model save dir
    model_save_dir = Path(config.training.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    best_test_loss = float("inf")
    test_loss_mean = float("inf")

    print(f"Job started at: {now()}")

    tglobal = trange(
        config.training.start_epoch,
        config.training.num_epochs,
        initial=config.training.start_epoch,
        total=config.training.num_epochs,
        desc=f"Epoch ({config.rollout.furniture if config.rollout.rollouts else 'multitask'}, {config.observation_type}, {config.vision_encoder.model})",
    )
    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        # batch loop
        actor.train_mode()
        dataset.train()
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

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            lr = lr_scheduler.get_last_lr()[0]
            wandb.log(
                dict(
                    lr=lr,
                    batch_loss=loss_cpu,
                )
            )

            tepoch.set_postfix(loss=loss_cpu, lr=lr)

        tepoch.close()

        train_loss_mean = np.mean(epoch_loss)

        # Evaluation loop
        actor.eval_mode()
        dataset.eval()
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

        # Early stopping
        if early_stopper.update(test_loss_mean):
            print(
                f"Early stopping at epoch {epoch_idx} as test loss did not improve for {early_stopper.patience} epochs."
            )
            break

        tglobal.set_postfix(
            time=now(),
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            stopper_counter=early_stopper.counter,
        )

        # Log epoch stats
        wandb.log(
            {
                "epoch_loss": np.mean(epoch_loss),
                "epoch": epoch_idx,
                "test_epoch_loss": test_loss_mean,
                "epoch": epoch_idx,
                "early_stopper/counter": early_stopper.counter,
                "early_stopper/best_loss": early_stopper.best_loss,
                "early_stopper/ema_loss": early_stopper.ema_loss,
            }
        )

        # Trigger sync at the end off all logging in the epoch
        trigger_sync()

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    main()
