#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96GB
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=02_ol_img_40_demos_unet

export HOME=/data/scratch/ankile

python -m src.train.bc +experiment=image/diff_unet \
    furniture=one_leg \
    rollout=rollout rollout.every=25 rollout.max_steps=700 \
    rollout.num_envs=64 rollout.count=256 \
    rollout.randomness=low \
    pred_horizon=32 action_horizon=8 obs_horizon=1 control.controller=diffik \
    demo_source=teleop randomness=low \
    training.encoder_lr=1e-5 \
    training.eval_every=5 training.sample_every=-1 \
    lr_scheduler.encoder_warmup_steps=50000 \
    data.suffix=demo_scaling \
    data.data_subset=40 \
    training.batch_size=256 training.actor_lr=1e-4 training.num_epochs=400 \
    training.steps_per_epoch=1000 \
    wandb.project=ol-vision-sim-demo-scaling-low-1 \
    wandb.name=ol-40-demos-unet-6 \
    wandb.continue_run_id=531a184b \
    dryrun=false
