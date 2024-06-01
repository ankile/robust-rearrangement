#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=6_rt_unet_lg_med

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[med,med_perturb]' \
    rollout.randomness=med \
    furniture=round_table \
    rollout.max_steps=1000 \
    wandb.continue_run_id=z3efusm6 \
    wandb.project=rt-state-dr-med-1 \
    dryrun=false