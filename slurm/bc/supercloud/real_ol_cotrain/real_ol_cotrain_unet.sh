#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_cotrain_unet
#SBATCH -c 20

# Run your command with the provided arguments
python -m src.train.bc +experiment=image/real_ol_cotrain \
    wandb.project=real-one_leg_cotrain-1 \
    wandb.mode=offline \
    dryrun=false
