#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 3-00:00
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_multitask \
    data.data_subset=200 \
    training.num_epochs=1000 \
    wandb.project=multitask-everything-1
