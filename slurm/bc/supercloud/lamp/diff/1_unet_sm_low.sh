#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_lp_unet_sm_low

python -m src.train.bc +experiment=state/diff_unet \
    actor.diffusion_model.down_dims='[64,128,256]' \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    rollout.max_steps=1000 \
    task=lamp \
    wandb.project=lp-state-dr-low-1 \
    dryrun=false