#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 3-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH --job-name=2_lp_cotrain_10

python -m src.train.bc +experiment=image/real_ol_cotrain \
    furniture=lamp \
    randomness='[low,med]' \
    environment='[real,sim]' \
    data.data_paths_override='[diffik/real/lamp/teleop/low/success.zarr,diffik/sim/lamp_render_rppo/rollout/low/success.zarr,diffik/sim/lamp_render_rppo/rollout/med/success.zarr,diffik/sim/lamp_render_demos_colors/teleop/med/success.zarr,diffik/sim/lamp_render_demos_colors/teleop/med_perturb/success.zarr,diffik/sim/lamp_render_rppo_colors/rollout/low/success.zarr,diffik/sim/lamp_render_rppo_colors/rollout/med/success.zarr]' \
    actor.confusion_loss_beta=0.0 \
    data.minority_class_power=false \
    obs_horizon=1 \
    +data.max_episode_count.lamp.teleop.low.success=10 \
    wandb.project=real-lamp-cotrain-2 \
    wandb.mode=offline \
    dryrun=false

