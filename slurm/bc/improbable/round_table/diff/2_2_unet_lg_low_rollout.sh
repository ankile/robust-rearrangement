#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=2_2_rt_low_rollout

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[low,low_perturb]' \
    demo_source='[teleop,rollout]' \
    rollout.randomness=low \
    task=round_table \
    rollout.max_steps=1000 \
    wandb.project=rt-state-dr-low-1 \
    dryrun=false