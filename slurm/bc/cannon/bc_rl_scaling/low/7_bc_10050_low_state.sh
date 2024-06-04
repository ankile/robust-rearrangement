#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=7_bc_10050_low_state

python -m src.train.bc +experiment=state/scaling_10k \
    observation_type=state \
    dryrun=false
