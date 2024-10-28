#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=1_fph_mlp_low

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness=low \
    wandb.name=mlp-lg-ch-1 \
    rollout.max_steps=200 \
    task=factory_peg_hole \
    wandb.project=fph-state-dr-low-1 \
    dryrun=false
