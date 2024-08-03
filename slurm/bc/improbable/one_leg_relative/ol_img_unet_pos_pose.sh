#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_img_pos_pose

export HOME=/data/scratch/ankile

python -m src.train.bc +experiment=image/diff_unet \
    actor.diffusion_model.down_dims='[128,256,512]' \
    randomness='[low,low_perturb]' \
    rollout=rollout rollout.randomness=low rollout.every=50 \
    rollout.max_steps=700 rollout.num_envs=32 \
    furniture=one_leg regularization.weight_decay=0 \
    training.ema.use=false \
    pred_horizon=32 training.batch_size=256 \
    training.encoder_lr=1e-5 \
    lr_scheduler.encoder_warmup_steps=50000 \
    actor.include_proprioceptive_pos=true \
    actor.include_proprioceptive_ori=true \
    actor.pose_pred_loss_coef=0.1 \
    control.control_mode=pos \
    training.clip_grad_norm=true \
    wandb.name=img-abs-pose-26 \
    wandb.project=ol-image-relative-1 \
    dryrun=false
