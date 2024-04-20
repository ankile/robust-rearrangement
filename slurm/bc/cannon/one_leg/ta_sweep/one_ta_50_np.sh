#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

# Launch the run
python -m src.train.bc_no_rollout \
    +experiment=image_traj_aug \
    furniture=one_leg \
    data.data_subset=50 \
    data.dataloader_workers=16 \
    data.pad_after=false \
    wandb.mode=offline

