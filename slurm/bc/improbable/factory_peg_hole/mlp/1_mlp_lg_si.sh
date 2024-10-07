#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=1_fph_mlp_lo_si

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness='[low]' \
    rollout.randomness=low \
    rollout.max_steps=200 \
    task=factory_peg_hole \
    wandb.name=mlp-lg-si-8 \
    pred_horizon=1 action_horizon=1 \
    wandb.project=fph-state-dr-low-1 \
    dryrun=false
