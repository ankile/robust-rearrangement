#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=1_mr_mlp_low_original

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness='[low]' \
    rollout.randomness=low \
    rollout.max_steps=400 \
    task=mug_rack \
    wandb.project=mr-state-dr-low-1 \
    dryrun=false
