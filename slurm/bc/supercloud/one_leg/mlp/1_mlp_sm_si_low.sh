#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=9_mlp_sm_si_low

python -m src.train.bc +experiment=state/mlp_sm_si \
    randomness='[low,low_perturb]' \
    task=one_leg \
    wandb.project=ol-state-dr-low-1 \
    wandb.mode=offline \
    dryrun=false