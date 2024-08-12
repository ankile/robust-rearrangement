#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 1-12:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=5_ol_cotrain

export OMP_NUM_THREADS=8

torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    furniture='[one_leg_full_new,one_leg_render_demos_brighter,one_leg_render_rppo_brighter]' \
    randomness='[low,med,med_perturb]' \
    training.clip_grad_norm=true \
    training.batch_size=128 \
    data.dataloader_workers=20 \
    environment='[real,sim]' \
    wandb.project=real-ol-demo-scaling-1 \
    wandb.name=ol-40-cotrain-5 \
    dryrun=false
