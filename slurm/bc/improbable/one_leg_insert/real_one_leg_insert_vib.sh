#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=real_ol_vib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-09:50
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=image/real_one_leg_insert task=one_leg_insert \
    training.ema.use=false training.ema.switch=false \
    vision_encoder=r3m vision_encoder.model=r3m_34 \
    vision_encoder.pretrained=true vision_encoder.freeze=false \
    regularization.feature_layernorm=true \
    regularization.vib_front_feature_beta=0.1 \
    dryrun=false
