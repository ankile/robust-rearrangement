#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=4_fph_tran_lg_low

python -m src.train.bc +experiment=state/diff_tran \
    actor/diffusion_model=transformer_big \
    randomness='[low]' \
    rollout.randomness=low \
    rollout.max_steps=200 \
    task=factory_peg_hole \
    wandb.name=trans-lg-1 \
    wandb.project=fph-state-dr-low-1 \
    dryrun=false
