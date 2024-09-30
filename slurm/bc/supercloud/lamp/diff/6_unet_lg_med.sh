#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=1_lp_diff_unet_sm_med

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[med,med_perturb]' \
    rollout.randomness=med \
    task=lamp \
    rollout.max_steps=1000 \
    wandb.project=lp-state-dr-med-1 \
    wandb.mode=offline \
    dryrun=false