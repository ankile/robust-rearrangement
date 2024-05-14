#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090
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
    pred_horizon=32 action_horizon=4 obs_horizon=1 control.controller=diffik \
    actor.diffusion_model.down_dims='[128,256,512]' \
    demo_source='[teleop,rollout]' randomness='[low,med]' \
    training.batch_size=2048 training.actor_lr=1e-4 training.num_epochs=2000 \
    dryrun=false