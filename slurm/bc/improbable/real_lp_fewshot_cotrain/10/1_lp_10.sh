#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_lp_10

python -m src.train.bc +experiment=image/real_ol_cotrain \
    task=lamp \
    randomness=low \
    environment=real \
    data.data_paths_override='[diffik/real/lamp/teleop/low/success.zarr]' \
    actor.confusion_loss_beta=0.0 \
    data.minority_class_power=false \
    obs_horizon=1 \
    +data.max_episode_count.lamp.teleop.low.success=10 \
    wandb.project=real-lamp-cotrain-2 \
    dryrun=false

