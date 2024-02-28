#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 1-08:00            
#SBATCH --mem=256G            
#SBATCH --gres=gpu:1          
#SBATCH -c 16           
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

python -m src.train.bc_no_rollout \
    +experiment=image_traj_aug \
    furniture=round_table \
    data.dataloader_workers=16 \
    data.pad_after=False \
    data.data_subset=100
