#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=6_2_rt_med_rollout

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[med,med_perturb]' \
    rollout.randomness=med \
    demo_source='[teleop,rollout]' \
    task=round_table \
    rollout.max_steps=1000 \
    wandb.project=rt-state-dr-med-1 \
    dryrun=false