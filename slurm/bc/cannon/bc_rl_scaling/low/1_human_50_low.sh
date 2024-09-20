#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=1_human_50_low

# Run vision-based training on one_leg task with low randomness
# with only the original 50 teleop demos as a baseline

python -m src.train.bc +experiment=image/scaling_low \
    data.data_paths_override='[diffik/sim/one_leg/teleop/low/success.zarr,diffik/sim/one_leg/teleop/low_perturb/success.zarr]' \
    training.num_epochs=5000 \
    training.actor_lr=5e-5 \
    wandb.continue_run_id=qkekpke9 \
    dryrun=false