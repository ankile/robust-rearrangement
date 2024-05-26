#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=1_mlp_sm_si
#SBATCH -c 20

python -m src.train.bc +experiment=state/mlp_sm_si \
    wandb.mode=offline \
    rollout.rollouts=false \
    dryrun=false