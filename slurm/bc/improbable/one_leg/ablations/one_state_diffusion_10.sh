#!/bin/bash

#SBATCH -p vision-pulkitag-v100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=00-12:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_diff_10

python -m src.train.bc +experiment=state/diff_unet task=one_leg \
    rollout=rollout rollout.every=25 rollout.max_steps=700 rollout.num_envs=512 \
    pred_horizon=32 action_horizon=8 obs_horizon=1 control.controller=diffik \
    demo_source=teleop randomness=low \
    training.batch_size=1024 training.actor_lr=1e-4 training.num_epochs=5000 \
    training.steps_per_epoch=100 \
    data.dataloader_workers=10 \
    data.data_subset=10 \
    wandb.project=ol-state-dr-1 \
    wandb.name=ol-10-demos-21 \
    dryrun=false