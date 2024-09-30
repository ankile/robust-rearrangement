#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ol_diff_adapt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1

# cd /data/scratch/ankile/robust-rearrangement
# git checkout diffusion-adapt

# cd /data/scratch/ankile/diffusion-adapt
# git checkout lars-task-tasks

# source env.sh

WANDB_ENTITY=ankile DIFF_BASE=/data/scratch/ankile WANDB_MODE=online python script/train_diffusion.py --config-name=finetune_ppo_diffusion_unet_task --config-dir=cfg/robomimic/ppo-ft-state
