#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=8_rl_100050_low_state

python -m src.train.bc +experiment=state/scaling_100k_rt \
    wandb.continue_run_id=h1u90k07 \
    dryrun=false
