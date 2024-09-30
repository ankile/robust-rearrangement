#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=6_ol_upwt_cf_e4_action_weight
#SBATCH -c 20


python -m src.train.bc +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    randomness='[low,med,med_perturb]' \
    task='[one_leg_render_rppo,one_leg_full,one_leg_highres]' \
    environment='[real,sim]' \
    actor.confusion_loss_beta=1e-4 \
    actor.rescale_loss_for_domain=true \
    actor.confusion_loss_anchored=false \
    actor.weight_confusion_loss_by_action=true \
    wandb.mode=offline \
    dryrun=false
