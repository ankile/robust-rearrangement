#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=11_diff_tran_sm_low

python -m src.train.bc +experiment=state/diff_tran \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    rollout.rollouts=false \
    wandb.project=ol-state-dr-low-1 \
    wandb.mode=offline \
    dryrun=false