#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=21_ol_cotrain_10_black_upwt_cf4
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=1e-4 \
    training.num_epochs=5000 \
    demo_source='[teleop,rollout]' \
    task='[one_leg_render_demos_brighter,one_leg_render_rppo_brighter,one_leg_render_demos_black,one_leg_render_rppo_black,one_leg_full]' \
    randomness='[low,med,med_perturb]' \
    actor.confusion_loss_beta=1e-4 \
    environment='[real,sim]' \
    +data.max_episode_count.one_leg_full.teleop.low.success=10 \
    data.data_subset=200 \
    wandb.mode=offline \
    dryrun=false
