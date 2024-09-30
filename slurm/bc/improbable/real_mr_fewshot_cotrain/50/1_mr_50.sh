#!/bin/bash

#SBATCH -p vision-pulkitag-h100,vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_mr_50

python -m src.train.bc +experiment=image/real_ol_cotrain \
    task=mug_rack \
    randomness=low \
    environment=real \
    data.data_paths_override='[diffik/real/mug_rack_handle/teleop/low/success.zarr]' \
    actor.confusion_loss_beta=0.0 \
    data.minority_class_power=false \
    obs_horizon=1 \
    wandb.project=real-mr-cotrain-1 \
    dryrun=false \

