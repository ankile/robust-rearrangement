#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=1_lp_diff_unet_sm

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    task=lamp \
    rollout.max_steps=1000 \
    wandb.project=lp-state-dr-low-1 \
    wandb.mode=offline \
    dryrun=false