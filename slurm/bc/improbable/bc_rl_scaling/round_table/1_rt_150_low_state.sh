#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_rt_150_low_state
#SBATCH --requeue

python -m src.train.bc +experiment=state/scaling/round_table/1k \
    data.data_subset=100 \
    wandb.name=rt-150-11 \
    wandb.continue_run_id=1f16dda6 \
    dryrun=false
