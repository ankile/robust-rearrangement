#!/bin/bash

#SBATCH -p vision-pulkitag-h100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=8_ol_cotrain_conf_2

python -m src.train.bc +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    task='[one_leg_full_new,one_leg_render_demos_brighter,one_leg_render_rppo_brighter]' \
    randomness='[low,med,med_perturb]' \
    training.clip_grad_norm=true \
    training.batch_size=256 \
    data.dataloader_workers=4 \
    environment='[real,sim]' \
    data.minority_class_power=3 \
    actor.confusion_loss_beta=1e-2 \
    wandb.project=real-ol-demo-scaling-1 \
    wandb.name=ol-40-cotrain-conf-e2-1 \
    dryrun=false

