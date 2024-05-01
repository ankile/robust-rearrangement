#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_curriculum_3 \
    training.load_checkpoint_run_id=round_table-curriculum-1/runs/y5s10vgj \
    training.actor_lr=1e-5 \
    furniture=round_table \
    data.dataloader_workers=16 \
    demo_source='[teleop, rollout]' \
    wandb.name=finetune-partial-noaug-3
