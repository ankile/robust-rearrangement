#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=00-12:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_state_noise_all_data

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[low,low_perturb,med,med_perturb,high,high_perturb]' \
    rollout.randomness=low rollout.every=50 \
    task=one_leg \
    rollout.max_steps=700 \
    wandb.project=ol-state-dr-noise-1 \
    training.batch_size=256 \
    regularization.state_noise=0.0 \
    dryrun=false