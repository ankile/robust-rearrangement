#!/bin/bash

#SBATCH -p vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH -t 1-08:00
#SBATCH --mem=64G            
#SBATCH --gres=gpu:1          
#SBATCH -c 32
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log

source /data/scratch/ankile/.config/.slurm.env
python -m src.train.bc_no_rollout \
    +experiment=image_baseline \
    furniture=one_leg \
    data.dataloader_workers=32 \
    vision_encoder.pretrained=false \
    wandb.name=one_bl_no_pretrain
