#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 0-16:00            
#SBATCH --mem=128G            
#SBATCH --gres=gpu:1          
#SBATCH -c 16       
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

python -m src.train.bc_no_rollout +experiment=image_baseline furniture=one_leg data.dataloader_workers=16
