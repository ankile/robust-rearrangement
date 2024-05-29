#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=8_diff_tran_lg_high

python -m src.train.bc +experiment=state/diff_tran \
    actor/diffusion_model=transformer_big \
    randomness='[high]' \
    data.data_subset=50 \
    rollout.randomness=high \
    rollout.rollouts=false \
    wandb.mode=offline \
    wandb.project=ol-state-dr-high-1 \
    dryrun=false