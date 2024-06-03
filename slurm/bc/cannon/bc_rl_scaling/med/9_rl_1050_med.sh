#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=9_rl_1050_med

# Run vision-based training on one_leg furniture with med randomness
# with the original 50 teleop demos plus 1000 rollout demos from RL training

python -m src.train.bc +experiment=image/diff_unet \
    data.data_paths_override='[diffik/sim/one_leg/teleop/med/success.zarr,diffik/sim/one_leg/teleop/med_perturb/success.zarr,diffik/sim/one_leg/rollout/med/success/rppo_med_000.zarr,diffik/sim/one_leg/rollout/med/success/rppo_med_250.zarr,diffik/sim/one_leg/rollout/med/success/rppo_med_500.zarr,diffik/sim/one_leg/rollout/med/success/rppo_med_750.zarr]' \
    dryrun=false
