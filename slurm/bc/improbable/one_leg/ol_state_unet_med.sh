#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ol_state_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=00-12:00
#SBATCH --gres=gpu:1

python -m src.train.bc +experiment=state/diffusion task=one_leg \
    rollout=rollout rollout.every=10 rollout.max_steps=1000 rollout.num_envs=256 \
    pred_horizon=32 action_horizon=8 obs_horizon=1 control.controller=diffik \
    demo_source='[teleop,rollout]' randomness='[med,med_perturb]' \
    training.batch_size=4096 training.actor_lr=1e-4 training.num_epochs=2000 \
    wandb.project=ol-state-dr-1 dryrun=false