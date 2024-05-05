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
python -m src.train.bc +experiment=state/diffusion furniture=one_leg dryrun=false \
    rollout=rollout rollout.every=5 rollout.max_steps=1000 rollout.num_envs=256 \
    pred_horizon=16 action_horizon=8 control.controller=diffik \
    demo_source='[teleop,rollout]'