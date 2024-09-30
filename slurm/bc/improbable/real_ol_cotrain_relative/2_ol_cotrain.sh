#!/bin/bash

#SBATCH -p vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:2
#SBATCH --job-name=2_r_ol_ct

export OMP_NUM_THREADS=4
export HOME=/data/scratch/ankile


torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    task='[one_leg_full,one_leg_render_rppo_brighter,one_leg_render_demos_brighter]' \
    randomness='[low,med,med_perturb]' \
    environment='[real,sim]' \
    training.batch_size=128 \
    data.dataloader_workers=$OMP_NUM_THREADS \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    wandb.mode=online \
    wandb.project=real-ol-cotrain-relative-1 \
    dryrun=false
