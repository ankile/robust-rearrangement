#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 3-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16

# Run vision-based training on one_leg furniture with low randomness
# with only the original 50 teleop demos as a baseline

python -m src.train.bc +experiment=image/diff_unet \
    rollout=rollout \
    demo_source=teleop \
    furniture=one_leg \
    randomness='[low,low_perturb]' \
    wandb.project=ol-vision-scaling-low-1 \
    dryrun=false