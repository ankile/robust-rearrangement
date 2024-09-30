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
    rollout=rollout rollout.every=10 rollout.max_steps=700 rollout.num_envs=512 \
    pred_horizon=32 action_horizon=8 obs_horizon=1 control.controller=diffik \
    training.batch_size=1024 training.actor_lr=1e-4 training.num_epochs=10000 \
    rollout.randomness=med \
    task=round_table \
    rollout.max_steps=1000 \
    training.steps_per_epoch=1000 \
    training.actor_lr=3e-4 \
    training.ema.use=true \
    wandb.project=rt-state-dr-med-1 \
    dryrun=false