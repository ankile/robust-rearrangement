#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 3-00:00
#SBATCH --gres=gpu:volta:2
#SBATCH -c 40
#SBATCH --job-name=1_ol_50_gpu

OMP_NUM_THREADS=20

torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=image/real_ol_cotrain \
    demo_source=teleop \
    task='[one_leg_full]' \
    randomness=low \
    environment=real \
    training.batch_size=128 \
    data.dataloader_workers=20 \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    wandb.mode=offline \
    wandb.project=real-ol-cotrain-relative-1 \
    dryrun=false

