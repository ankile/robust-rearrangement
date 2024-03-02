#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-16:00
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_traj_aug_infer_ep_limit \
    furniture=lamp \
    data.dataloader_workers=16 \
    data.pad_after=False \
    data.data_subset=400
