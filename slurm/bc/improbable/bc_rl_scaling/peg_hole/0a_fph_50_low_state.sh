#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=0a_fph_50_low_state
#SBATCH --requeue

python -m src.train.bc +experiment=state/scaling/peg_hole/base \
    wandb.name=fph-bc-50-10 \
    wandb.continue_run_id=fcecbdf2 \
    dryrun=false
