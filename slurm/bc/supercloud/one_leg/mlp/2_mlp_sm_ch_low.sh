#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=10_mlp_sm_ch_low

python -m src.train.bc +experiment=state/mlp_sm_ch \
    randomness='[low,low_perturb]' \
    wandb.project=ol-state-dr-low-1 \
    task=one_leg \
    rollout.rollouts=false \
    wandb.mode=offline \
    dryrun=false