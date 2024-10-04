#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=7_rl_50050_low_state

python -m src.train.bc +experiment=state/scaling_50k_rt \
    wandb.continue_run_id=rwx4fn8m \
    dryrun=false
