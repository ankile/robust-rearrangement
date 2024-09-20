#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 0-12:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=1_ol_10_demos

python -m src.train.bc +experiment=image/real_ol_cotrain \
    demo_source=teleop \
    task=one_leg \
    randomness=low \
    environment=real \
    training.batch_size=256 \
    data.dataloader_workers=16 \
    data.data_paths_override='[diffik/real/one_leg_full_new/teleop/low/success.zarr]' \
    training.clip_grad_norm=true \
    +data.max_episode_count.one_leg_full_new.teleop.low.success=10 \
    wandb.project=real-ol-demo-scaling-1 \
    wandb.name=ol-10-demos-1 \
    wandb.continue_run_id=u804edpk \
    dryrun=false