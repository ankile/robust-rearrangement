#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=12_diff_tran_lg_low

python -m src.train.bc +experiment=state/diff_tran \
    actor/diffusion_model=transformer_big \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    wandb.project=ol-state-dr-low-1 \
    rollout.rollouts=false \
    wandb.mode=offline \
    dryrun=false