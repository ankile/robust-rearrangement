#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=8_2_rt_mlp_lg_ch_med_rollout

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness='[med,med_perturb]' \
    demo_source='[teleop,rollout]' \
    rollout.randomness=med \
    task=round_table \
    rollout.max_steps=1000 \
    wandb.project=rt-state-dr-med-1 \
    dryrun=false