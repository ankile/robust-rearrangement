#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Run your command with the provided arguments
python -m src.train.bc +experiment=image/real_one_leg_insert \
    task=one_leg_insert \
    vision_encoder=r3m vision_encoder.model=r3m_18 \
    vision_encoder.pretrained=true vision_encoder.freeze=false \
    dryrun=false