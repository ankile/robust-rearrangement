#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=3_mlp_lg_si
#SBATCH -c 20

python -m src.train.bc +experiment=state/mlp_lg_si \
    wandb.mode=offline \
    dryrun=false