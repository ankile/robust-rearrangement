#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=5_diff_unet_sm_high
#SBATCH -c 20

python -m src.train.bc +experiment=state/diff_unet \
    actor.diffusion_model.down_dims='[64,128,256]' \
    randomness='[high]' \
    data.data_subset=50 \
    rollout.randomness=high \
    wandb.project=ol-state-dr-high-1 \
    wandb.mode=offline \
    dryrun=false