#!/bin/bash

#SBATCH -p seas_gpu
#SBATCH -t 4-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=0b_lp_1k_low_state

python -m src.train.bc +experiment=state/scaling/lamp/1k \
    data.data_subset=100 \
    wandb.name=lp-150-1 \
    dryrun=false
