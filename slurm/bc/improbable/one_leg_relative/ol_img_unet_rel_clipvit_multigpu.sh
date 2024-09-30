#!/bin/bash

#SBATCH -p vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=ol_img_diff_rel_clipvit

export OMP_NUM_THREADS=16

torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=image/diff_unet \
    actor.diffusion_model.down_dims='[128,256,512]' \
    vision_encoder=clip_vit \
    rollout=rollout rollout.randomness=low rollout.every=50 \
    randomness='[low,low_perturb]' \
    task=one_leg regularization.weight_decay=0 \
    training.ema.use=false \
    training.encoder_lr=1e-5 \
    pred_horizon=32 training.batch_size=64 \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    wandb.watch_model=false \
    wandb.name=img-rel-clipvit-25 \
    wandb.project=ol-image-relative-1 \
    dryrun=true
