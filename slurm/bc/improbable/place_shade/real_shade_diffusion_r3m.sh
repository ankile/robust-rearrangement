#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-h100
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=real_shade_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-16:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=image/real_place_shade_r3m task=place_shade \
    data.normalization=none dryrun=false training.ema.use=false training.ema.switch=false \
    vision_encoder.pretrained=true vision_encoder.freeze=true