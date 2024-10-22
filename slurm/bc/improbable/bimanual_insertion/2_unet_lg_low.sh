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
#SBATCH --job-name=2_bi_unet_lg_low


python -m src.train.bc +experiment=state/diff_unet \
    randomness=low \
    task=bimanual_insertion \
    rollout.num_envs=1 \
    rollout.count=16 \
    pred_horizon=64 \
    action_horizon=16 \
    control.controller=dexhub \
    wandb.project=bi-state-dr-low-1 \
    dryrun=false
