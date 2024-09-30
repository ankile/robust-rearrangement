#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 3-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
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
    wandb.mode=offline \
    dryrun=false

