#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 3-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_multitask_everything \
    training.num_epochs=1000 \
    data.dataloader_workers=16 \
    furniture='[lamp,round_table]' \
    wandb.name=mt-unet-big-2tasks-rn-1
