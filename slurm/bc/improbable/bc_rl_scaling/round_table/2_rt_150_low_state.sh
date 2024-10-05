#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-v100,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=4_rt_150_low_state

python -m src.train.bc +experiment=image/scaling_low \
    data.data_paths_override='[diffik/sim/round_table/teleop/low/success.zarr,diffik/distillation/round_table/rollout/low/success/rppo_0.zarr]' \
    observation_type=state \
    furniture=round_table \
    rollout.num_envs=512 rollout.max_steps=1000 \
    wandb.name=rt-150-old-2 \
    data.data_subset=100 \
    wandb.project=rt-scaling-low-1 \
    dryrun=false
