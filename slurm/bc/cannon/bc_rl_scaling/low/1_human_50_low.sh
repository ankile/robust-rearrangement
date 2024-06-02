#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=1_human_50_low

# Run vision-based training on one_leg furniture with low randomness
# with only the original 50 teleop demos as a baseline

python -m src.train.bc +experiment=image/diff_unet \
    rollout=rollout \
    rollout.num_envs=128 \
    rollout.every=50 \
    demo_source=teleop \
    furniture=one_leg \
    randomness='[low,low_perturb]' \
    wandb.project=ol-vision-scaling-low-1 \
    dryrun=false