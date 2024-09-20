#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_img_diff_rel_clipvit

export HOME=/data/scratch/ankile

python -m src.train.bc +experiment=image/diff_unet \
    actor.diffusion_model.down_dims='[128,256,512]' \
    vision_encoder=clip_vit \
    vision_encoder.freeze=true \
    randomness='[low,low_perturb]' \
    rollout=rollout rollout.randomness=low rollout.every=50 \
    rollout.max_steps=700 rollout.num_envs=32 \
    task=one_leg regularization.weight_decay=0 \
    training.ema.use=false \
    pred_horizon=32 training.batch_size=256 \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    training.eval_every=50 \
    wandb.watch_model=true \
    wandb.name=img-rel-clipvit-15 \
    wandb.project=ol-image-relative-1 \
    wandb.continue_run_id=44ef7014 \
    dryrun=false
