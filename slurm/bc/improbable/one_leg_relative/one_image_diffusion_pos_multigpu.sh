#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_img_pos_baseline

torchrun --standalone --nproc_per_node=1 -m src.train.bc_ddp +experiment=image/diff_unet \
    actor.diffusion_model.down_dims='[128,256,512]' \
    training.batch_size=64 \
    randomness='[low,low_perturb]' \
    task=one_leg regularization.weight_decay=0 \
    rollout=rollout rollout.randomness=low rollout.every=1 \
    rollout.max_steps=700 rollout.num_envs=32 \
    training.ema.use=false \
    training.clip_grad_norm=true \
    control.control_mode=pos \
    wandb.name=img-baseline-clip-1 \
    wandb.watch_model=false \
    wandb.project=ol-image-relative-1 \
    dryrun=true

#     rollout=rollout rollout.randomness=low rollout.every=5 \
#     rollout.max_steps=700 rollout.num_envs=32 \

#     distributed=true \

#     actor.diffusion_model.down_dims='[128,256,512]' \
