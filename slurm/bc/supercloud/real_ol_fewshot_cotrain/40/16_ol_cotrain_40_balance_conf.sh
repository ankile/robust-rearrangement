#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=16_ol_cotrain_40_balance_conf
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    demo_source='[teleop,rollout]' \
    furniture='[one_leg_render_demos_brighter,one_leg_render_rppo_brighter,one_leg_full]' \
    randomness='[low,med,med_perturb]' \
    environment='[real,sim]' \
    +data.max_episode_count.one_leg_full.teleop.low.success=40 \
    data.minority_class_power=3 \
    actor.confusion_loss_beta=1e-4 \
    wandb.mode=offline \
    dryrun=false