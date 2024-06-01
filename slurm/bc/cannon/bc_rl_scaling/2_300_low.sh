#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=1_diff_bc_rollouts
#SBATCH -c 20

python -m src.train.bc +experiment=image/diff_unet \
    demo_source=teleop \
    furniture=one_leg \
    wandb.mode=offline \
    rollout.rollouts=true \
    rollout.randomness=low \
    data.data_paths_override='[diffik/sim/one_leg/teleop/low/success.zarr,diffik/sim/one_leg/teleop/low_perturb/success.zarr]' \
    dryrun=false