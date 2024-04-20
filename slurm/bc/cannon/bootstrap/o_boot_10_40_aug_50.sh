#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

python -m src.train.bc_no_rollout \
    +experiment=image_bootstrap \
    training.load_checkpoint_run_id=null \
    furniture=one_leg \
    demo_source='[teleop, rollout, augmentation]' \
    wandb.name=one_bootstrap-10-40-aug-50 \
    data.max_episode_count.one_leg.rollout.low.success=40 \
    data.max_episode_count.one_leg.augmentation.low.success=50

