#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_sim_trans
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=5e-4 \
    environment=sim \
    demo_source='[teleop,rollout]' \
    training.num_epochs=5000 \
    actor.confusion_loss_beta=0.0 \
    wandb.mode=offline \
    dryrun=false
