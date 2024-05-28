#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=3_diff_tran_sm
#SBATCH -c 20

python -m src.train.bc +experiment=state/diff_tran \
    randomness='[high,high_perturb]' \
    data.data_subset=25 \
    rollout.randomness=high \
    wandb.mode=offline \
    wandb.project=ol-state-dr-high-1 \
    dryrun=false