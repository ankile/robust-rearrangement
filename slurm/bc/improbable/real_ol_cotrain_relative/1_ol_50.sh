#!/bin/bash

#SBATCH -p vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=1_ol_50_gpu

export OMP_NUM_THREADS=16

torchrun --standalone --nproc_per_node=4 -m src.train.bc_ddp +experiment=image/real_ol_cotrain \
    demo_source=teleop \
    task='[one_leg_full]' \
    randomness=low \
    environment=real \
    training.batch_size=64 \
    data.dataloader_workers=$OMP_NUM_THREADS \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    wandb.mode=online \
    wandb.project=real-ol-cotrain-relative-1 \
    dryrun=false
