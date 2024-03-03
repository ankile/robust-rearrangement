#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 1-16:00            
#SBATCH --mem=128G            
#SBATCH --gres=gpu:1          
#SBATCH -c 16              
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

python -m src.train.bc_no_rollout \
    +experiment=image_collect_infer \
    furniture=one_leg \
    data.data_subset=50 \
    data.dataloader_workers=16 \
    vision_encoder=r3m \
    wandb.name=ci-50-r3m-1
