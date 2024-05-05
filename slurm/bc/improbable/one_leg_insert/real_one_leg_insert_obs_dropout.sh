#!/bin/bash

#SBATCH -p vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-h100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=real_ol_dropout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-16:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=image/real_one_leg_insert furniture=one_leg_insert \
    training.ema.use=false training.ema.switch=false \
    vision_encoder=r3m vision_encoder.pretrained=true vision_encoder.freeze=false \
    regularization.front_camera_dropout=0.9 regularization.proprioception_dropout=0.05 \
    dryrun=false