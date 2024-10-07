#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=0c_fph_100_low_state
#SBATCH --requeue

python -m src.train.bc +experiment=state/scaling/peg_hole/1k \
    data.data_subset=50 \
    wandb.name=fph-rl-100-11 \
    wandb.continue_run_id=d4748f50 \
    dryrun=false
