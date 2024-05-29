#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=12_mlp_lg_ch_low

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness='[low,low_perturb]' \
    rollout.randomness=low \
    wandb.project=ol-state-dr-low-1 \
    dryrun=false