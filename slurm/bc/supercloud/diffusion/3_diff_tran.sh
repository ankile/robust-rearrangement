#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=1_diff_tran
#SBATCH -c 20

python -m src.train.bc +experiment=state/diff_tran \
    wandb.mode=offline \
    dryrun=false