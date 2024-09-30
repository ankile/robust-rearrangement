#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-h100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=3_only_camera_1_freeze

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    vision_encoder.freeze=true \
    training.actor_lr=1e-4 \
    training.encoder_lr=1e-6 \
    training.num_epochs=5000 \
    demo_source=teleop \
    task=pick_cup \
    regularization.front_camera_dropout=1.0 \
    randomness=low \
    environment=real \
    wandb.entity=dexterity-hub \
    wandb.project=pick-cup-1 \
    wandb.name=one-camera-freeze-1 \
    dryrun=false