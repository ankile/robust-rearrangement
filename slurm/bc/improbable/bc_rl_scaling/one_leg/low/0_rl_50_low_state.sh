#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-v100,vision-pulkitag-a100,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=4_rl_1050_low_state

python -m src.train.bc +experiment=image/scaling_low \
    data.data_paths_override='[diffik/sim/one_leg/teleop/low/success.zarr,diffik/sim/one_leg/teleop/low_perturb/success.zarr]' \
    observation_type=state \
    dryrun=false
