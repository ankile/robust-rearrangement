import os
from pathlib import Path
import furniture_bench
from furniture_bench.sim_config import sim_config
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from src.dataset.dataset import (
    FurnitureImageDataset,
    FurnitureFeatureDataset,
)
from src.dataset.normalizer import StateActionNormalizer
from src.eval.rollout import do_rollout_evaluation
from src.common.tasks import furniture2idx, task_timeout
from src.gym import get_env
from tqdm import tqdm, trange
from ipdb import set_trace as bp
from src.behavior import get_actor
from src.dataset.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_apply
import argparse
from torch.utils.data import random_split, DataLoader
from src.common.earlystop import EarlyStopper
from src.common.control import RotationMode, ControlMode
from src.common.files import get_processed_paths


from ml_collections import ConfigDict

from gym import logger

logger.set_level(logger.DISABLED)


def main(config: ConfigDict, start_epoch: int = 0):
    env = None
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if config.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_paths=config.data_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            augment_image=config.augment_image,
            data_subset=config.data_subset,
            first_action_idx=config.first_action_index,
        )
    elif config.observation_type == "feature":
        dataset = FurnitureFeatureDataset(
            dataset_paths=config.data_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            encoder_name=config.vision_encoder.model,
            data_subset=config.data_subset,
            first_action_idx=config.first_action_index,
        )
    else:
        raise ValueError(f"Unknown observation type: {config.observation_type}")

    # Split the dataset into train and test
    train_size = int(len(dataset) * (1 - config.test_split))
    test_size = len(dataset) - train_size
    print(f"Splitting dataset into {train_size} train and {test_size} test samples.")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Update the config object with the action dimension
    config.action_dim = dataset.action_dim
    config.robot_state_dim = dataset.robot_state_dim
    config.n_episodes = len(dataset.episode_ends)

    # Create the policy network
    actor = get_actor(config, device)

    # Update the config object with the observation dimension
    config.timestep_obs_dim = actor.timestep_obs_dim

    if config.load_checkpoint_path is not None:
        print(f"Loading checkpoint from {config.load_checkpoint_path}")
        actor.load_state_dict(torch.load(config.load_checkpoint_path))

    # Create dataloaders
    trainloader = FixedStepsDataloader(
        dataset=train_dataset,
        n_batches=config.steps_per_epoch,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    # AdamW optimizer for noise_net
    opt_noise = torch.optim.AdamW(
        params=actor.parameters(),
        lr=config.actor_lr,
        weight_decay=config.weight_decay,
    )

    n_batches = len(trainloader)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.name,
        optimizer=opt_noise,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * config.num_epochs,
    )

    early_stopper = EarlyStopper(
        patience=config.early_stopper.patience,
        smooth_factor=config.early_stopper.smooth_factor,
    )

    # Init wandb
    wandb.init(
        id=config.wandb.continue_run_id,
        resume=config.wandb.continue_run_id is not None,
        project=config.wandb.project,
        entity="robot-rearrangement",
        config=config.to_dict(),
        mode=config.wandb.mode if not config.dryrun else "disabled",
        notes=config.wandb.notes,
    )

    # save stats to wandb and update the config object
    wandb.log(
        {
            "num_samples": len(train_dataset),
            "num_samples_test": len(test_dataset),
            "num_episodes": int(len(dataset.episode_ends) * (1 - config.test_split)),
            "num_episodes_test": int(len(dataset.episode_ends) * config.test_split),
            "stats": StateActionNormalizer().stats_dict,
        }
    )
    wandb.config.update(config.to_dict())

    # Create model save dir
    model_save_dir = Path(config.model_save_dir) / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    best_test_loss = float("inf")
    test_loss_mean = float("inf")
    best_success_rate = 0

    tglobal = trange(
        start_epoch,
        config.num_epochs,
        initial=start_epoch,
        total=config.num_epochs,
        desc=f"Epoch ({config.furniture if 'furniture' in config else 'multitask'}, {config.observation_type}, {config.vision_encoder.model})",
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
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Get loss
            loss = actor.compute_loss(batch)

            # backward pass
            loss.backward()

            # Gradient clipping
            if config.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=config.clip_grad_norm
                )

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
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        # Evaluation loop
        actor.eval_mode()
        dataset.eval()
        test_tepoch = tqdm(testloader, desc="Validation", leave=False)
        for test_batch in test_tepoch:
            with torch.no_grad():
                # device transfer for test_batch
                test_batch = dict_apply(
                    test_batch, lambda x: x.to(device, non_blocking=True)
                )

                # Get test loss
                test_loss_val = actor.compute_loss(test_batch)

                # logging
                test_loss_cpu = test_loss_val.item()
                test_loss.append(test_loss_cpu)
                test_tepoch.set_postfix(loss=test_loss_cpu)

        test_tepoch.close()

        test_loss_mean = np.mean(test_loss)
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_success_rate,
        )

        wandb.log({"test_epoch_loss": test_loss_mean, "epoch": epoch_idx})

        # Save the model if the test loss is the best so far
        if config.checkpoint_model and test_loss_mean < best_test_loss:
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

        # Log the early stopping stats
        wandb.log(
            {
                "early_stopper/counter": early_stopper.counter,
                "early_stopper/best_loss": early_stopper.best_loss,
                "early_stopper/ema_loss": early_stopper.ema_loss,
            }
        )

        if (
            config.rollout.every != -1
            and (epoch_idx + 1) % config.rollout.every == 0
            and np.mean(test_loss_mean) < config.rollout.loss_threshold
        ):
            # Do not load the environment until we successfuly made it this far
            if env is None:
                env: FurnitureSimEnv = get_env(
                    config.gpu_id,
                    furniture=config.furniture,
                    num_envs=config.num_envs,
                    randomness=config.randomness,
                    # Now using full size images in sim and resizing to be consistent
                    resize_img=False,
                    act_rot_repr=config.act_rot_repr,
                    ctrl_mode="osc",
                )

            best_success_rate = do_rollout_evaluation(
                config,
                env,
                config.save_rollouts,
                actor,
                best_success_rate,
                epoch_idx,
            )

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--dryrun", "-d", action="store_true")
    parser.add_argument("--cpus", "-c", type=int, default=24)
    parser.add_argument("--wb-mode", "-w", type=str, default="online")
    parser.add_argument(
        "--obs-type", type=str, default="image", choices=["image", "feature"]
    )
    parser.add_argument("--encoder", "-e", type=str, default="vip")
    parser.add_argument("--furniture", "-f", type=str, default=None)
    parser.add_argument("--no-rollout", action="store_true")
    parser.add_argument("--data-subset", type=int, default=None)
    parser.add_argument("--n-parts-assemble", type=int, default=None)

    args = parser.parse_args()

    dryrun = lambda x, fb=1: x if args.dryrun is False else fb

    n_workers = min(args.cpus, os.cpu_count())
    num_envs = dryrun(8, fb=2)

    config = ConfigDict()

    config.wandb = ConfigDict()
    # config.wandb.project = "image-training"
    # config.wandb.project = "new-controller-test"
    # config.wandb.project = "encoder-comparison"
    config.wandb.project = "teleop-finetune"
    # config.wandb.project = "simple-regularization"
    config.wandb.notes = "Train on precomputed data, all."
    config.wandb.mode = args.wb_mode
    config.wandb.continue_run_id = None

    # defaults
    config.action_horizon = 8
    config.pred_horizon = 24
    config.first_action_index = 0
    config.obs_horizon = 2

    config.actor = "diffusion"

    # RNN options
    # config.actor = "rnn"
    # config.action_horizon = 1
    # config.first_action_index = -1  # aligns with the final observation in the sequence
    # config.obs_horizon = 10

    # MLP options
    # config.actor_hidden_dims = [4096, 4096, 2048]
    # config.actor_dropout = 0.2

    # Diffusion options
    config.beta_schedule = "squaredcos_cap_v2"
    # config.down_dims = [128, 256, 512]
    config.down_dims = [256, 512, 1024]
    # config.down_dims = [512, 1024, 2048]
    config.inference_steps = 16
    config.prediction_type = "epsilon"
    config.num_diffusion_iters = 100

    config.save_rollouts = False
    config.actor_lr = 1e-4
    config.batch_size = args.batch_size
    config.clip_grad_norm = False
    config.data_subset = dryrun(args.data_subset, 5)
    config.dataloader_workers = n_workers
    config.clip_sample = True
    config.demo_source = "sim"
    config.dryrun = args.dryrun

    # TODO: Change this to "task"
    config.furniture = args.furniture
    config.n_parts_assemble = args.n_parts_assemble

    config.gpu_id = args.gpu_id
    config.load_checkpoint_path = None
    # config.load_checkpoint_path = "/data/scratch/ankile/furniture-diffusion/models/ruby-butterfly-17/actor_chkpt_best_test_loss.pt"
    config.mixed_precision = False
    config.num_envs = num_envs
    config.num_epochs = 200
    config.observation_type = args.obs_type
    config.randomness = "low"
    config.steps_per_epoch = dryrun(400, fb=10)
    config.test_split = 0.05

    config.rollout = ConfigDict()
    config.rollout.every = dryrun(5, fb=1) if not args.no_rollout else -1
    if not args.no_rollout:
        config.rollout.loss_threshold = dryrun(0.05, fb=float("inf"))
        config.rollout.max_steps = dryrun(
            task_timeout(config.furniture, n_parts=args.n_parts_assemble),
            fb=100,
        )
        config.rollout.count = num_envs * 1
        config.rollout.save_failures = False

    config.lr_scheduler = ConfigDict()
    config.lr_scheduler.name = "cosine"
    config.lr_scheduler.warmup_steps = 500

    config.vision_encoder = ConfigDict()
    config.vision_encoder.model = args.encoder
    config.vision_encoder.freeze = True
    config.vision_encoder.pretrained = True
    # config.vision_encoder.encoding_dim = 256
    # config.vision_encoder.normalize_features = False

    config.early_stopper = ConfigDict()
    config.early_stopper.smooth_factor = 0.9
    config.early_stopper.patience = 10

    config.discount = 0.999

    # Multi-task options
    config.multi_task = True
    config.task_dim = 16
    config.num_tasks = len(furniture2idx)
    config.language_conditioning = False

    # Trajectory success conditioning options
    config.trajectory_success_conditioning = False

    # Regularization
    config.weight_decay = 1e-6

    config.feature_dropout = False
    # config.feature_dropout = 0.1

    config.feature_noise = False
    # config.feature_noise = 0.01

    # config.feature_layernorm = False
    config.feature_layernorm = True

    config.augment_image = False
    # config.augmentation = ConfigDict()
    # config.augmentation.translate = 10
    # config.augmentation.color_jitter = False

    config.model_save_dir = "models"
    config.checkpoint_model = True

    # Control mode
    config.act_rot_repr = RotationMode.rot_6d
    config.control_mode = ControlMode.delta

    assert (
        config.rollout.every == -1 or config.rollout.count % config.num_envs == 0
    ), "n_rollouts must be divisible by num_envs"

    config.data_path = get_processed_paths(
        environment="sim",
        task=None,
        # task=config.furniture,
        demo_source=["teleop", "scripted"],
        randomness=None,
        demo_outcome="success",
    )

    print(f"Using data from {config.data_path}")

    main(config, start_epoch=0)
