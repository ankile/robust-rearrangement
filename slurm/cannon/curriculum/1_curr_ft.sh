#!/bin/bash

#SBATCH -p gpu                
#SBATCH -t 1-00:00            
#SBATCH --mem=256G            
#SBATCH --gres=gpu:1          
#SBATCH -c 32                 
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log   

python -m src.train.bc_no_rollout \
    +experiment=image_curriculum_1 \
    training.load_checkpoint_run_id=round_table-collectinfer-1/runs/p2r481d1 \
    furniture=round_table \
    wandb.name=finetune-partial-2
