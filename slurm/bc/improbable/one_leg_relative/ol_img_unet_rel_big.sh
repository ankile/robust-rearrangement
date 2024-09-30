#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_img_diff_rel_clip_big

python -m src.train.bc +experiment=image/diff_unet \
    actor.diffusion_model.down_dims='[256,512,1024]' \
    randomness='[low,low_perturb]' \
    rollout=rollout rollout.randomness=low rollout.every=50 \
    rollout.max_steps=700 rollout.num_envs=32 \
    task=one_leg regularization.weight_decay=1e-6 \
    training.ema.use=false \
    pred_horizon=32 training.batch_size=256 \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    wandb.watch_model=false \
    wandb.name=img-rel-clip-big-12 \
    wandb.project=ol-image-relative-1 \
    wandb.continue_run_id=hecj9w5n \
    dryrun=false
    