#!/bin/bash

#SBATCH -p vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH -t 1-16:00
#SBATCH --mem=200G            
#SBATCH --gres=gpu:1          
#SBATCH -c 16 
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log

source /data/scratch/ankile/.config/.slurm.env

python -m src.train.bc_no_rollout \
    +experiment=image_curriculum_3 \
    training.load_checkpoint_run_id=round_table-curriculum-1/runs/y5s10vgj \
    training.actor_lr=1e-5 \
    furniture=round_table \
    data.dataloader_workers=16 \
    wandb.name=finetune-partial-aug-3

