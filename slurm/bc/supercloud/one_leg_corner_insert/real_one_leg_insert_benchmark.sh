#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=bc_benchmark_new
#SBATCH -c 20

git checkout actor-action-chunk-refactor

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
    randomness='[low]' \
    wandb.project=real-image-speed-compare-channels-first \
    wandb.mode=offline \
    wandb.name=bc-benchmark-new-1 \
    dryrun=false
