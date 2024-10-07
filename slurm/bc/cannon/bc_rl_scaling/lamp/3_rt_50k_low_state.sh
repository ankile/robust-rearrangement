#!/bin/bash

#SBATCH -p seas_gpu
#SBATCH -t 7-00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=3_lp_50k_low_state

python -m src.train.bc +experiment=state/scaling/lamp/50k \
    wandb.name=lp-50k-1 \
    dryrun=false
