#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=3_ol_color_40_upsamp

python -m src.train.bc +experiment=image/real_ol_cotrain_color \
    furniture=one_leg \
    randomness='[low,med]' \
    environment='[real,sim]' \
    actor.confusion_loss_beta=0.0 \
    data.minority_class_power=3 \
    obs_horizon=1 \
    wandb.mode=offline \
    dryrun=false

