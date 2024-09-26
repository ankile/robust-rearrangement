#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-v100,vision-pulkitag-a100,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=4_rl_1050_low_state

python -m src.train.bc +experiment=state/scaling_1k \
    wandb.continue_run_id=659bc152 \
    dryrun=false
