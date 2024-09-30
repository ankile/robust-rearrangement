#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 3-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=3_ol_color_40_upsamp

python -m src.train.bc +experiment=image/real_ol_cotrain_color \
    task=one_leg \
    randomness='[low,med]' \
    environment='[real,sim]' \
    actor.confusion_loss_beta=0.0 \
    data.minority_class_power=3 \
    obs_horizon=1 \
    wandb.mode=offline \
    dryrun=false

