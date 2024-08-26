#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=2_diff_unet_lg

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    furniture=round_table \
    rollout.max_steps=700 \
    wandb.project=ol-state-dr-low-1 \
    regularization.weight_decay=1e-5 \
    dryrun=false