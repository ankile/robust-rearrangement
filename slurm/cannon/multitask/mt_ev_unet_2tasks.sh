#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 3-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_multitask \
    training.num_epochs=1000 \
    data.dataloader_workers=16 \
    furniture='[lamp,round_table]' \
    data.pad_after=false \
    wandb.project=multitask-2tasks-1
