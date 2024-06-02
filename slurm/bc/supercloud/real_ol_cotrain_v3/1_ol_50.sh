#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=1_ol_50
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    demo_source=teleop \
    furniture='[one_leg_full]' \
    randomness=low \
    environment=real \
    wandb.mode=offline \
    dryrun=false

