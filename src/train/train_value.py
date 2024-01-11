import os
from pathlib import Path
import numpy as np
import torch
import wandb
from diffusers.optimization import get_scheduler
from src.dataset.dataset import OfflineRLFeatureDataset
from src.dataset.normalizer import StateActionNormalizer
from tqdm import tqdm
from ipdb import set_trace as bp
from src.models.value import CriticModule
from src.dataset.dataloader import FixedStepsDataloader
from src.common.pytorch_util import dict_apply
import argparse
from torch.utils.data import random_split, DataLoader
from src.common.earlystop import EarlyStopper

from ml_collections import ConfigDict


def main(config: ConfigDict):
    env = None
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if config.observation_type == "feature":
        dataset = OfflineRLFeatureDataset(
            dataset_path=config.data_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            normalizer=StateActionNormalizer(),
            data_subset=config.data_subset,
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
    config.obs_dim = config.robot_state_dim + 2 * dataset.feature_dim

    # Create the critic module
    critic = CriticModule(
        action_dim=config.action_dim,
        action_horizon=config.action_horizon,
        obs_dim=config.obs_dim,
        obs_horizon=config.obs_horizon,
        critic_dropout=config.critic_dropout,
        critic_hidden_dims=config.critic_hidden_dims,
        discount=config.discount,
        expectile=config.expectile,
        device=device,
    )

    if config.load_checkpoint_path is not None:
        print(f"Loading checkpoint from {config.load_checkpoint_path}")
        critic.load_state_dict(torch.load(config.load_checkpoint_path))

    # Create dataloaders
    trainloader = FixedStepsDataloader(
        dataset=train_dataset,
        n_batches=config.steps_per_epoch,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    testloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    # AdamW optimizer for noise_net
    opt_noise = torch.optim.AdamW(
        params=critic.parameters(),
        lr=config.critic_lr,
        weight_decay=config.weight_decay,
    )

    n_batches = len(trainloader)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.name,
        optimizer=opt_noise,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        num_training_steps=len(trainloader) * config.num_epochs,
    )

    tglobal = tqdm(range(config.num_epochs), desc="Epoch")
    best_test_loss = float("inf")

    early_stopper = EarlyStopper(
        patience=config.early_stopper.patience,
        smooth_factor=config.early_stopper.smooth_factor,
    )

    # Init wandb
    wandb.init(
        project="critic-module-test",
        entity="robot-rearrangement",
        config=config.to_dict(),
        mode="online" if not config.dryrun else "disabled",
        notes="Train the value function module",
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
    model_save_dir = Path(config.model_save_dir) / "critic_module" / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    test_loss_mean = 0.0
    for epoch_idx in tglobal:
        epoch_loss = list()
        test_loss = list()

        # batch loop
        tepoch = tqdm(trainloader, desc="Training", leave=False, total=n_batches)
        critic.train()
        for batch in tepoch:
            opt_noise.zero_grad()

            # device transfer
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Get loss
            q_loss, v_loss = critic.compute_loss(batch)
            total_loss = q_loss + v_loss

            # backward pass
            total_loss.backward()

            # optimizer step
            opt_noise.step()
            lr_scheduler.step()

            critic.polyak_update_target(config.q_target_update_step)

            # logging
            loss_cpu = total_loss.item()
            epoch_loss.append(loss_cpu)
            lr = lr_scheduler.get_last_lr()[0]
            wandb.log(
                dict(
                    lr=lr,
                    batch_loss=loss_cpu,
                    batch_q_loss=q_loss.item(),
                    batch_v_loss=v_loss.item(),
                )
            )

            tepoch.set_postfix(
                loss=loss_cpu,
                lr=lr,
                q_loss=q_loss.item(),
                v_loss=v_loss.item(),
            )

        tepoch.close()

        train_loss_mean = np.mean(epoch_loss)
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
        )
        wandb.log({"epoch_loss": np.mean(epoch_loss), "epoch": epoch_idx})

        # Evaluation loop
        test_tepoch = tqdm(testloader, desc="Validation", leave=False)

        # Make list for storing the mean v and q values to track over time to ensure they don't diverge
        v_values = []
        q_values = []

        critic.eval()
        for test_batch in test_tepoch:
            with torch.no_grad():
                # device transfer for test_batch
                test_batch = dict_apply(
                    test_batch, lambda x: x.to(device, non_blocking=True)
                )

                # Get test loss
                q_loss, v_loss = critic.compute_loss(test_batch)
                test_loss_val = q_loss + v_loss

                # Get the mean q and v values
                nobs = critic._training_obs(test_batch["curr_obs"])
                naction = critic._flat_action(test_batch["action"])
                q = critic.q_value(nobs, naction)
                v = critic.value(nobs)

                # Append the mean q and v values to the list
                q_values.append(q.mean().item())
                v_values.append(v.mean().item())

                # logging
                test_loss_cpu = test_loss_val.item()
                test_loss.append(test_loss_cpu)
                test_tepoch.set_postfix(
                    loss=test_loss_cpu,
                    q_loss=q_loss.item(),
                    v_loss=v_loss.item(),
                )

        test_tepoch.close()

        test_loss_mean = np.mean(test_loss)
        tglobal.set_postfix(
            loss=train_loss_mean,
            test_loss=test_loss_mean,
            best_success_rate=best_test_loss,
            # Log the mean q and v values
            mean_q_value=np.mean(q_values),
            mean_v_value=np.mean(v_values),
        )

        wandb.log({
            "test_epoch_loss": test_loss_mean, "epoch": epoch_idx,
            "mean_q_value": np.mean(q_values),
            "mean_v_value": np.mean(v_values),
        })

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

        if config.checkpoint_model and test_loss_mean < best_test_loss:
            # Checkpoint the model
            save_path = str(model_save_dir / f"actor_chkpt_latest.pt")
            torch.save(
                critic.state_dict(),
                save_path,
            )
            wandb.save(save_path)

        # Update the best test loss seen
        best_test_loss = min(best_test_loss, test_loss_mean)

    tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--dryrun", "-d", action="store_true")
    parser.add_argument("--cpus", "-c", type=int, default=24)
    parser.add_argument("--wb-mode", "-w", type=str, default="online")
    args = parser.parse_args()

    maybe = lambda x, fb=1: x if args.dryrun is False else fb

    n_workers = min(args.cpus, os.cpu_count())
    num_envs = maybe(8, fb=2)

    config = ConfigDict()

    config.action_horizon = 8
    config.pred_horizon = 16

    config.data_base_dir = Path(os.environ.get("FURNITURE_DATA_DIR_PROCESSED", "data"))
    config.batch_size = args.batch_size
    config.clip_grad_norm = False
    config.data_subset = None if args.dryrun is False else 10
    config.dataloader_workers = n_workers
    config.demo_source = "sim"
    config.dryrun = args.dryrun
    config.furniture = "one_leg"
    config.gpu_id = args.gpu_id
    config.load_checkpoint_path = None
    config.num_epochs = 200
    config.obs_horizon = 2
    config.observation_type = "feature"
    config.test_split = 0.05
    config.steps_per_epoch = 500

    config.lr_scheduler = ConfigDict()
    config.lr_scheduler.name = "cosine"
    config.lr_scheduler.warmup_steps = 500

    config.vision_encoder = ConfigDict()
    config.vision_encoder.model = "vip"

    config.early_stopper = ConfigDict()
    config.early_stopper.smooth_factor = 0.9
    config.early_stopper.patience = float("inf")

    config.discount = 0.997

    # Regularization
    config.weight_decay = 1e-6

    # Q-learning
    config.expectile = 0.75
    config.q_target_update_step = 0.005
    config.discount = 0.997
    config.critic_dropout = 0.3
    config.critic_lr = 1e-4
    config.critic_weight_decay = 1e-6
    config.critic_hidden_dims = [2048, 1024, 1024, 512]

    config.model_save_dir = "models"
    config.checkpoint_model = True

    # config.data_path = "/data/scratch/ankile/furniture-data/data/processed/sim/feature_small/vip/one_leg/data_new.zarr"

    # For experimentation, use a dataset with only the last n time steps for each episode and terminal states added
    config.data_path = (
        "/data/scratch/ankile/furniture-diffusion/notebooks/data_short.zarr"
    )
    
    # For experimentation, use a dataset with artificially dense rewards and terminal states
    # config.data_path = (
    #     "/data/scratch/ankile/furniture-diffusion/notebooks/data_dense.zarr"
    # )

    # For experimentation, use a dataset with skill boundaries as rewards as well as terminal states
    # config.data_path = (
    #     "/data/scratch/ankile/furniture-diffusion/notebooks/data_skill.zarr"
    # )    

    # For experimentation, use a dataset with terminal states added
    # config.data_path = (
    #     "/data/scratch/ankile/furniture-diffusion/notebooks/data_terminal.zarr"
    # )

    print(f"Using data from {config.data_path}")

    main(config)
