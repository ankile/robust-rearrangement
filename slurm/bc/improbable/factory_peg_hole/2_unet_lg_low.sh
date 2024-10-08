#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=2_fph_unet_lg_low

python -m src.train.bc +experiment=state/diff_unet \
    randomness=low \
    rollout.max_steps=200 rollout.num_envs=1024 rollout.every=1 \
    training.steps_per_epoch=500 \
    pred_horizon=8 action_horizon=4 \
    task=factory_peg_hole \
    wandb.name=unet-lg-10 \
    wandb.project=fph-state-dr-low-1 \
    dryrun=false
