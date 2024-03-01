#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 3-00:00
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_baseline \
    furniture=square_table \
    data.dataloader_workers=16 \
    vision_encoder=r3m \
    wandb.name=square_bl_r3m
