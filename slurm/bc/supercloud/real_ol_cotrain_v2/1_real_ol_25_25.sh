#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=2_r_ol_25_25
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    training.actor_lr=1e-4 \
    training.num_epochs=2000 \
    demo_source=teleop \
    task='[one_leg_full,one_leg_simple]' \
    randomness=low \
    environment=real \
    data.data_subset=25 \
    wandb.mode=offline \
    dryrun=false

