#!/bin/bash

#SBATCH -p vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH -t 1-08:00
#SBATCH --mem=128G            
#SBATCH --gres=gpu:1          
#SBATCH -c 32
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log

source /data/scratch/ankile/.config/.slurm.env

python -m src.train.bc_no_rollout \
    +experiment=image_bootstrap \
    training.load_checkpoint_run_id=null \
    furniture=one_leg \
    demo_source='[teleop, rollout]' \
    wandb.name=one_bootstrap-10-90-noaug \
    data.max_episode_count.rollout.low.success=40

