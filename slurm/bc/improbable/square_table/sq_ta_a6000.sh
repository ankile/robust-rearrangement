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
    +experiment=image_traj_aug \
    furniture=square_table \
    data.dataloader_workers=32 \
    data.pad_after=True \
    data.data_subset=250
