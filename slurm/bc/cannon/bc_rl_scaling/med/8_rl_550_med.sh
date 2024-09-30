#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=8_550_med

# Run vision-based training on one_leg task with med randomness
# with the original 50 teleop demos plus 500 rollout demos from RL training

python -m src.train.bc +experiment=image/scaling_med \
    data.data_paths_override='[diffik/sim/one_leg/teleop/med/success.zarr,diffik/sim/one_leg/teleop/med_perturb/success.zarr,diffik/sim/one_leg/rollout/med/success/rppo_med_000.zarr,diffik/sim/one_leg/rollout/med/success/rppo_med_250.zarr]' \
    training.actor_lr=5e-5 \
    wandb.continue_run_id=oyd7q63s \
    dryrun=false
