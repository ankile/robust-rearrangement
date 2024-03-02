#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-16:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_traj_aug_infer_one_leg \
    furniture=one_leg \
    data.data_subset=100 \
    data.dataloader_workers=16 \
    wandb.name=taci-100-aug-1

