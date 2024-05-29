#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=11_mlp_lg_si_low

python -m src.train.bc +experiment=state/mlp_lg_si \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    wandb.project=ol-state-dr-low-1 \
    rollout.rollouts=false \
    wandb.mode=offline \
    dryrun=false