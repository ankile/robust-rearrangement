#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=5_r_ol_ct_cf0001_tran
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=1e-4 \
    demo_source='[teleop,rollout]' \
    training.num_epochs=5000 \
    task='[one_leg_render_rppo,one_leg_simple,one_leg_full]' \
    randomness='[low,med,med_perturb]' \
    actor.confusion_loss_beta=0.001 \
    environment='[real,sim]' \
    wandb.mode=offline \
    dryrun=false
