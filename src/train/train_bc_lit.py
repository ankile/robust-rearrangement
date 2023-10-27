import os
from pathlib import Path
import furniture_bench
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from src.data.dataset import FurnitureImageDataset, SimpleFurnitureDataset
from src.eval.callbacks import RolloutEvaluationCallback
from src.gym import get_env
from tqdm import tqdm
from ipdb import set_trace as bp
from src.models.actor_lit import LitImageActor
from src.common.pytorch_util import dict_apply
import argparse
import lightning.pytorch as pl

from ml_collections import ConfigDict
from lightning.pytorch.loggers import WandbLogger

model_save_dir = Path("models")


def main(config: ConfigDict):
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Init wandb
    wandb.init(
        project="furniture-diffusion",
        entity="ankile",
        config=config.to_dict(),
        mode="online" if not config.dryrun else "disabled",
    )
    config = wandb.config

    if config.observation_type == "image":
        dataset = FurnitureImageDataset(
            dataset_path=config.datasim_path,
            pred_horizon=config.pred_horizon,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            data_subset=config.data_subset,
        )
    else:
        raise ValueError(f"Unknown observation type: {config.observation_type}")

    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # Update the config object with the action dimension
    config.action_dim = dataset.action_dim
    config.robot_state_dim = dataset.robot_state_dim

    # Create the policy network
    actor = LitImageActor(
        encoder_name="resnet18",
        config=config,
        stats=stats,
    )

    # Update the config object with the observation dimension
    config.obs_dim = actor.obs_dim

    # save stats to wandb
    wandb.log(
        {
            "num_samples": len(dataset),
            "num_episodes": len(dataset.episode_ends),
            "stats": stats,
        }
    )

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.dataloader_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    wandb_logger = WandbLogger()

    callback = RolloutEvaluationCallback(config, get_env)

    trainer = pl.Trainer(
        devices=[0, 1, 2, 3],
        accelerator="gpu",
        max_epochs=config.num_epochs,
        check_val_every_n_epoch=20,
        precision=16,
        logger=wandb_logger,
        callbacks=[callback],
    )
    trainer.fit(actor, dataloader)

    # tglobal.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--dryrun", "-d", action="store_true")
    parser.add_argument("--cpus", "-c", type=int, default=24)
    args = parser.parse_args()

    data_base_dir = Path(os.environ.get("FURNITURE_DATA_DIR", "data"))
    maybe = lambda x, fb=1: x if args.dryrun is False else fb

    config = ConfigDict(
        dict(
            action_horizon=8,
            actor_lr=1e-4,
            batch_size=args.batch_size,
            beta_schedule="squaredcos_cap_v2",
            clip_grad_norm=False,
            clip_sample=True,
            dataloader_workers=args.cpus,
            demo_source="sim",
            down_dims=[256, 512, 1024],
            dryrun=args.dryrun,
            furniture="one_leg",
            gpu_id=args.gpu_id,
            inference_steps=16,
            lr_scheduler_type="cosine",
            lr_scheduler_warmup_steps=500,
            mixed_precision=False,
            n_rollouts=10 if args.dryrun is False else 1,
            num_diffusion_iters=100,
            num_envs=1,  # Use 1 env for now
            num_epochs=200,
            obs_horizon=2,
            observation_type="image",
            pred_horizon=16,
            prediction_type="epsilon",
            randomness="high",
            rollout_every=20 if args.dryrun is False else 1,
            rollout_loss_threshold=1e9,
            rollout_max_steps=750 if args.dryrun is False else 10,
            vision_encoder_pretrained=False,
            vision_encoder="resnet18",
            weight_decay=1e-6,
            data_subset=None if args.dryrun is False else 10,
        )
    )

    assert (
        config.n_rollouts % config.num_envs == 0
    ), "n_rollouts must be divisible by num_envs"

    config.datasim_path = data_base_dir / "processed/sim/image/one_leg/high/data.zarr"

    print(f"Using data from {config.datasim_path}")

    main(config)
