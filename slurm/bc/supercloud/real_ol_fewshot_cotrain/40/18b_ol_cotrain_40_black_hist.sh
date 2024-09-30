#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 3-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=18b_ol_cotrain_40_black_hist
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    demo_source='[teleop,rollout]' \
    task='[one_leg_render_demos_brighter,one_leg_render_rppo_brighter,one_leg_render_demos_black,one_leg_render_rppo_black,one_leg_full]' \
    randomness='[low,med,med_perturb]' \
    environment='[real,sim]' \
    +data.max_episode_count.one_leg_full.teleop.low.success=40 \
    data.data_subset=200 \
    actor.confusion_loss_beta=0.0 \
    obs_horizon=2 \
    wandb.mode=offline \
    dryrun=true
