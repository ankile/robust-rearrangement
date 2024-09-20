#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=12_mlp_lg_ch_high

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness='[high]' \
    data.data_subset=50 \
    rollout.randomness=high \
    wandb.project=ol-state-dr-high-1 \
    wandb.mode=offline \
    task=one_leg \
    rollout.rollouts=false \
    dryrun=false