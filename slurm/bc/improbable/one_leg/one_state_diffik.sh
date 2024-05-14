#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-v100,vision-pulkitag-h100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=oneleg_state_diffik
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=state/diffusion furniture=one_leg \
    rollout=rollout rollout.every=20 rollout.max_steps=1000 rollout.num_envs=256 \
    pred_horizon=32 action_horizon=8 obs_horizon=1 control.controller=diffik \
    demo_source='[teleop,rollout]' randomness='[low,med]' \
    training.batch_size=4096 training.actor_lr=5e-4 training.num_epochs=2000 \
    dryrun=false