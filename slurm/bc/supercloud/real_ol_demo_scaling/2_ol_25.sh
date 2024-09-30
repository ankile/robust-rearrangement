#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 1-12:00
#SBATCH --gres=gpu:volta:2
#SBATCH -c 40
#SBATCH --job-name=2_ol_25_demos

export OMP_NUM_THREADS=20

torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=image/real_ol_cotrain \
    demo_source=teleop \
    task=one_leg \
    randomness=low \
    environment=real \
    training.batch_size=128 \
    data.dataloader_workers=20 \
    data.data_paths_override='[diffik/real/one_leg_full_new/teleop/low/success.zarr]' \
    training.clip_grad_norm=true \
    +data.max_episode_count.one_leg_full_new.teleop.low.success=25 \
    wandb.mode=offline \
    wandb.project=real-ol-demo-scaling-1 \
    wandb.name=ol-25-demos-1 \
    dryrun=false