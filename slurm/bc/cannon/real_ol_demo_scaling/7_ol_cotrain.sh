#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-12:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=7_ol_cotrain

python -m src.train.bc +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    task='[one_leg_full_new,one_leg_render_demos_brighter,one_leg_render_rppo_brighter]' \
    randomness='[low,med,med_perturb]' \
    training.clip_grad_norm=true \
    training.batch_size=256 \
    data.dataloader_workers=16 \
    environment='[real,sim]' \
    wandb.project=real-ol-demo-scaling-1 \
    wandb.name=ol-40-cotrain-1 \
    dryrun=false
