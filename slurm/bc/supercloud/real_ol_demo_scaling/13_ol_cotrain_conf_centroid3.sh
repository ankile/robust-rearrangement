#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 1-12:00
#SBATCH --gres=gpu:volta:2
#SBATCH -c 40
#SBATCH --job-name=13_ol_cotrain_conf_centroid3

export OMP_NUM_THREADS=20

torchrun --standalone --nproc_per_node=2 -m src.train.bc_ddp +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    task='[one_leg_full_new,one_leg_render_demos_brighter,one_leg_render_rppo_brighter]' \
    randomness='[low,med,med_perturb]' \
    training.clip_grad_norm=true \
    training.batch_size=128 \
    data.dataloader_workers=20 \
    environment='[real,sim]' \
    data.minority_class_power=3 \
    actor.confusion_loss_beta=1e-3 \
    actor.confusion_loss_centroid_formulation=true \
    wandb.mode=offline \
    wandb.project=real-ol-demo-scaling-1 \
    wandb.name=ol-40-cotrain-cf-centroid-e3-1 \
    dryrun=false
