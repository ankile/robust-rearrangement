#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 0-15:00            
#SBATCH --mem=256G            
#SBATCH --gres=gpu:1          
#SBATCH -c 32                 
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

# Run the wandb agent command
python -m src.train.bc_no_rollout +experiment=image_multitask multitask=multitask 'actor.diffusion_model.down_dims=[512, 1024, 2048]'
# python -m src.train.bc_no_rollout +experiment=image_traj_aug furniture=square_table
# python -m src.train.bc_no_rollout +experiment=image_baseline furniture=square_table
# python -m src.train.bc_no_rollout +experiment=image_collect_infer furniture=square_table
python -m src.train.bc_no_rollout +experiment=image_cherry_noise furniture=round_table
