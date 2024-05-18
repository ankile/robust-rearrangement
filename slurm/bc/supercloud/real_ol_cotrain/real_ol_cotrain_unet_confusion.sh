#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_cotrain_unet_confusion
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor.confusion_loss_beta=0.1 \
    wandb.mode=offline \
    dryrun=false
