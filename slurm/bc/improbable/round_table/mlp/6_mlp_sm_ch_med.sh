#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=6_rt_mlp_sm_ch_med

python -m src.train.bc +experiment=state/mlp_sm_ch \
    randomness='[med,med_perturb]' \
    rollout.randomness=med \
    furniture=round_table \
    wandb.project=rt-state-dr-med-1 \
    dryrun=false