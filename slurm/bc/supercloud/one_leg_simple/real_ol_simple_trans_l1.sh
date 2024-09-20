#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 1-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_simple_trans_l1_mae_ema
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_one_leg_insert \
    vision_encoder=spatial_softmax \
    vision_encoder.pretrained=false vision_encoder.freeze=false vision_encoder.use_groupnorm=true \
    regularization.feature_layernorm=true \
    regularization.front_camera_dropout=0.1 \
    regularization.wrist_camera_dropout=0.0 \
    regularization.proprioception_dropout=0.0 \
    regularization.vib_front_feature_beta=0.0 \
    actor/diffusion_model=transformer \
    actor.loss_fn=L1Loss \
    training.actor_lr=1e-5 training.encoder_lr=1e-5 \
    training.num_epochs=5000 training.batch_size=256 \
    early_stopper.patience=inf \
    training.ema.use=true \
    task=one_leg_simple \
    environment=real \
    wandb.project=real-one_leg_simple-1 \
    wandb.mode=offline \
    dryrun=false
