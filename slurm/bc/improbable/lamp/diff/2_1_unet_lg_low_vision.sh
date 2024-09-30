#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=2_lp_unet_lg_low_vision

python -m src.train.bc +experiment=image/diff_unet \
    task=lamp \
    randomness='[low,low_perturb]' \
    rollout=rollout \
    rollout.randomness=low \
    rollout.max_steps=1000 \
    rollout.num_envs=32 \
    wandb.project=lp-state-dr-low-1 \
    dryrun=false