#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=2_1_rt_low_vision

python -m src.train.bc +experiment=image/diff_unet \
    task=round_table \
    randomness='[low,low_perturb]' \
    rollout=rollout \
    rollout.randomness=low \
    rollout.max_steps=1000 \
    rollout.num_envs=32 \
    wandb.continue_run_id=ji9yh1bo \
    wandb.project=rt-state-dr-low-1 \
    dryrun=false