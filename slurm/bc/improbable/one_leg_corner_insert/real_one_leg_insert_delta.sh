#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=real_olci_r3m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=0-16:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=image/real_one_leg_insert \
    vision_encoder=r3m vision_encoder.model=r3m_18 \
    vision_encoder.pretrained=true vision_encoder.freeze=false \
    regularization.feature_layernorm=true \
    regularization.front_camera_dropout=0.0 \
    regularization.wrist_camera_dropout=0.0 \
    regularization.proprioception_dropout=0.0 \
    task=one_leg_corner_insert \
    environment=real \
    control.control_mode=delta \
    actor.loss_fn=L1Loss \
    wandb.project=real-one_leg_corner_insert-1 \
    dryrun=false