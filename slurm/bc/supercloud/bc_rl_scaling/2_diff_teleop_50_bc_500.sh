#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=1_diff_bc_rollouts
#SBATCH -c 20

python -m src.train.bc +experiment=image/diff_unet \
    demo_source='[teleop,rollout]' \
    task=one_leg \
    randomness='[med,med_perturb]' \
    data.suffix='[rppo_2_000,rppo_2_250]' \
    data.data_subset=1000 \
    wandb.mode=offline \
    rollout.rollouts=false \
    dryrun=false