#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-h100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=512GB
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:2
#SBATCH --job-name=12_ol_cotrain_more_upwt

export OMP_NUM_THREADS=10

torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    task='[one_leg_full_new,one_leg_render_demos_brighter,one_leg_render_rppo_brighter,one_leg_render_rppo_1]' \
    randomness='[low,med,med_perturb]' \
    training.clip_grad_norm=true \
    training.batch_size=128 \
    data.dataloader_workers=10 \
    environment='[real,sim]' \
    wandb.mode=offline \
    data.minority_class_power=3 \
    wandb.project=real-ol-demo-scaling-1 \
    wandb.name=ol-40-cotrain-more-upwt-1 \
    dryrun=false
