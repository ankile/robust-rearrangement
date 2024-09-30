#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=11_mlp_lg_si_low

python -m src.train.bc +experiment=state/mlp_lg_si \
    randomness='[low,low_perturb]' \
    wandb.project=ol-state-dr-low-1 \
    task=one_leg \
    rollout.rollouts=false \
    wandb.mode=offline \
    dryrun=false