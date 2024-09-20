#!/bin/bash

#SBATCH -p vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_img_diff_rel_clip_frozen

python -m src.train.bc +experiment=image/diff_unet \
    actor.diffusion_model.down_dims='[128,256,512]' \
    vision_encoder.freeze=true \
    randomness='[low,low_perturb]' \
    rollout=rollout rollout.randomness=low rollout.every=50 \
    rollout.max_steps=700 rollout.num_envs=32 \
    task=one_leg regularization.weight_decay=0 \
    training.ema.use=false \
    pred_horizon=32 training.batch_size=256 \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    wandb.watch_model=false \
    wandb.name=img-rel-clip-frozen-1 \
    wandb.project=ol-image-relative-1 \
    dryrun=false
