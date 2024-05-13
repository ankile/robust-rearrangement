#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-h100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=ol_state_diff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=00-08:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=state/diffusion furniture=one_leg \
    rollout=rollout rollout.every=10 rollout.max_steps=1000 rollout.num_envs=256 \
    pred_horizon=16 action_horizon=8 obs_horizon=1 control.controller=diffik \
    demo_source='[teleop,rollout]' randomness='[low,med]' training.num_epochs=2000 \
    early_stopper.patience=inf training.batch_size=1024 training.steps_per_epoch=-1 \
    dryrun=false