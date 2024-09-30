#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=384G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=5_bc_1050_low

# Run vision-based training on one_leg task with low randomness
# with the original 50 teleop demos plus 1000 rollout demos from BC training

python -m src.train.bc +experiment=image/diff_unet \
    rollout=rollout \
    rollout.num_envs=128 \
    rollout.every=50 \
    task=one_leg \
    data.data_paths_override='[diffik/sim/one_leg/teleop/low/success.zarr,diffik/sim/one_leg/teleop/low_perturb/success.zarr,diffik/sim/one_leg/rollout/low/success/bc_low_000.zarr,diffik/sim/one_leg/rollout/low/success/bc_low_250.zarr,diffik/sim/one_leg/rollout/low/success/bc_low_500.zarr,diffik/sim/one_leg/rollout/low/success/bc_low_750.zarr]' \
    wandb.project=ol-vision-scaling-low-1 \
    dryrun=false
