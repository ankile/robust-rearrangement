#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_10-50_real-sim
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=1e-4 \
    training.num_epochs=2000 \
    demo_source=teleop \
    task='[one_leg_highres]' \
    randomness='[med,med_perturb]' \
    environment='[sim]' \
    wandb.mode=offline \
    dryrun=false

