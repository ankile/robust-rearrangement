#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=0b_fph_150_low_state
#SBATCH --requeue

python -m src.train.bc +experiment=state/scaling/peg_hole/1k \
    data.data_subset=100 \
    wandb.name=fph-rl-150-h16-26 \
    wandb.continue_run_id=e42ff465 \
    dryrun=false
