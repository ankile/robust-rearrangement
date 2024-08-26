#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=2_lp_cotrain_40

python -m src.train.bc +experiment=image/real_ol_cotrain \
    furniture=lamp \
    randomness='[low,med]' \
    environment='[real,sim]' \
    data.data_paths_override='[diffik/real/lamp/teleop/low/success.zarr,diffik/sim/lamp_render_rppo/rollout/low/success.zarr,diffik/sim/lamp_render_rppo/rollout/med/success.zarr,diffik/sim/lamp_render_demos_colors/teleop/med/success.zarr,diffik/sim/lamp_render_demos_colors/teleop/med_perturb/success.zarr,diffik/sim/lamp_render_rppo_colors/rollout/low/success.zarr,diffik/sim/lamp_render_rppo_colors/rollout/med/success.zarr]' \
    actor.confusion_loss_beta=0.0 \
    data.minority_class_power=false \
    obs_horizon=1 \
    wandb.project=real-lamp-cotrain-2 \
    dryrun=false

