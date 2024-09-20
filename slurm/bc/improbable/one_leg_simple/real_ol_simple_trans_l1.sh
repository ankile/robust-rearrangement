#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH -c 20
#SBATCH -t 1-00:00
#SBATCH --job-name=real_ol_simple_trans_l1

python -m src.train.bc +experiment=image/real_one_leg_insert \
    vision_encoder=r3m vision_encoder.model=r3m_18 \
    vision_encoder.pretrained=true vision_encoder.freeze=false \
    regularization.feature_layernorm=true \
    regularization.front_camera_dropout=0.0 \
    regularization.wrist_camera_dropout=0.0 \
    regularization.proprioception_dropout=0.0 \
    regularization.vib_front_feature_beta=0.0 \
    actor/diffusion_model=transformer \
    actor.loss_fn=L1Loss \
    training.actor_lr=1e-4 training.encoder_lr=1e-5 \
    training.num_epochs=5000 \
    early_stopper.patience=inf \
    task=one_leg_simple \
    environment=real \
    wandb.project=real-one_leg_simple-1 \
    wandb.mode=online \
    dryrun=false