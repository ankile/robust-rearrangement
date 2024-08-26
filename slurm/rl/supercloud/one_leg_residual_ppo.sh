#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=one_leg_residual_ppo
#SBATCH -c 20

python -m src.train.residual_ppo \
    wandb.mode=offline \
    debug=false
