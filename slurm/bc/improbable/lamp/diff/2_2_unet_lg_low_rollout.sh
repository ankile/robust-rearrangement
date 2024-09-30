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
#SBATCH --job-name=2_2_lp_low_rollout

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[low,low_perturb]' \
    demo_source='[teleop,rollout]' \
    rollout.randomness=low \
    task=lamp \
    rollout.max_steps=1000 \
    wandb.continue_run_id=hwj7kk4i \
    wandb.project=lp-state-dr-low-1 \
    dryrun=false
