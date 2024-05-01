#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 3-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_traj_aug_infer \
    data.data_subset=400 \
    data.pad_after=True \
    furniture=square_table
