#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32GB
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=ol_attentionpool_diffusion_unet_low

python -m src.train.bc +experiment=image/attentionpool_diffusion \
    task=one_leg \
    rollout=rollout rollout.every=10 rollout.max_steps=700 rollout.num_envs=16 \
    rollout.randomness=low \
    pred_horizon=32 action_horizon=8 obs_horizon=1 control.controller=diffik \
    demo_source=teleop randomness='[low,low_perturb]' \
    training.batch_size=256 training.actor_lr=1e-4 training.num_epochs=10000 \
    training.steps_per_epoch=1000 \
    wandb.project=attention-pool \
    dryrun=false