#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=2_r_ol_ct_10
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    furniture=lamp \
    data.data_paths_override='[diffik/real/lamp/teleop/low/success.zarr,diffik/sim/lamp_render_rppo/rollout/low/success.zarr]' \    randomness='[low,med,med_perturb]' \
    actor.confusion_loss_beta=0.0 \
    +data.max_episode_count.one_leg_full.teleop.low.success=10 \
    wandb.mode=offline \
    dryrun=false
