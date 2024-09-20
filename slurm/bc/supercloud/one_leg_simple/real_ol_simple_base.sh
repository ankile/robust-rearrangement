#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_simple_base
#SBATCH -c 20

# Run your command with the provided arguments
python -m src.train.bc +experiment=image/real_one_leg_insert \
    vision_encoder=r3m vision_encoder.model=r3m_18 \
    vision_encoder.pretrained=true vision_encoder.freeze=false \
    regularization.feature_layernorm=true \
    regularization.front_camera_dropout=0.0 \
    regularization.wrist_camera_dropout=0.0 \
    regularization.proprioception_dropout=0.0 \
    regularization.vib_front_feature_beta=0.0 \
    actor/diffusion_model=unet \
    actor.diffusion_model.down_dims='[128,256,512]' \
    actor.loss_fn=L1Loss \
    early_stopper.patience=inf \
    training.actor_lr=1e-4 training.encoder_lr=1e-5 \
    task=one_leg_simple \
    environment=real \
    wandb.project=real-one_leg_simple-1 \
    wandb.mode=offline \
    dryrun=false
