#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=9_mlp_sm_si_high

python -m src.train.bc +experiment=state/mlp_sm_si \
    randomness='[high]' \
    data.data_subset=50 \
    wandb.project=ol-state-dr-high-1 \
    task=one_leg \
    rollout.rollouts=false \
    wandb.mode=offline \
    dryrun=false