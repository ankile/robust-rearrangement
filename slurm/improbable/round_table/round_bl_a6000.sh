#!/bin/bash

#SBATCH -p vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH -t 1-16:00
#SBATCH --mem=120G            
#SBATCH --gres=gpu:1          
#SBATCH -c 32
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log

source /data/scratch/ankile/.config/.slurm.env

python -m src.train.bc_no_rollout \
    +experiment=image_baseline \
    furniture=round_table \
    data.dataloader_workers=32
