#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=8_rt_tran_lg_med

python -m src.train.bc +experiment=state/diff_tran \
    actor/diffusion_model=transformer_big \
    randomness='[med,med_perturb]' \
    rollout.randomness=med \
    task=round_table \
    wandb.project=rt-state-dr-med-1 \
    dryrun=false