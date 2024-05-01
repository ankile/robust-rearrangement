#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-16:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_curriculum_2 \
    training.load_checkpoint_run_id=round_table-curriculum-1/runs/9ch9t9xp \
    furniture=round_table \
    wandb.name=finetune-partial-2 \
    data.dataloader_workers=16
