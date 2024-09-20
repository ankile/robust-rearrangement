#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=2_2_lp_low_rollout

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[low,low_perturb]' \
    demo_source='[teleop,rollout]' \
    rollout.randomness=low \
    task=lamp \
    rollout.max_steps=1000 \
    wandb.project=lp-state-dr-low-1 \
    dryrun=false
