#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=4b_ol_color_40_hist3

python -m src.train.bc +experiment=image/real_ol_cotrain_color \
    task=one_leg \
    randomness='[low,med]' \
    environment='[real,sim]' \
    actor.confusion_loss_beta=0.0 \
    data.minority_class_power=false \
    obs_horizon=3 \
    training.batch_size=128 \
    wandb.mode=offline \
    dryrun=false

