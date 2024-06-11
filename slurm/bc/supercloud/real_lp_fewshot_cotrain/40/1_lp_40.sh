#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 3-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=1_lp_40
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    furniture=lamp \
    data.data_paths_override='[diffik/real/lamp/teleop/low/success.zarr]' \
    wandb.mode=offline \
    dryrun=false \

