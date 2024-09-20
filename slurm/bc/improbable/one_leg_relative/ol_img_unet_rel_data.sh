#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_img_rel_data

export HOME=/data/scratch/ankile

python -m src.train.bc +experiment=image/diff_unet \
    actor.diffusion_model.down_dims='[128,256,512]' \
    randomness='[low,low_perturb,med,med_perturb]' \
    demo_source='[teleop,rollout]' \
    training.encoder_lr=1e-5 \
    lr_scheduler.encoder_warmup_steps=100000 \
    rollout=rollout rollout.randomness=low rollout.every=50 \
    rollout.max_steps=700 rollout.num_envs=32 \
    task=one_leg \
    training.ema.use=false \
    pred_horizon=32 training.batch_size=256 \
    control.control_mode=relative \
    wandb.project=ol-image-relative-1 \
    wandb.name=img-rel-data-24 \
    wandb.continue_run_id=h2sgfv04 \
    dryrun=false
