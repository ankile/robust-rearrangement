#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=2_bi_tran_lg_low

python -m src.train.bc +experiment=state/diff_tran \
    randomness=low \
    task=bimanual_insertion \
    control.controller=dexhub \
    rollout.num_envs=128 \
    rollout.max_steps=400 \
    rollout.count=256 \
    pred_horizon=64 \
    action_horizon=16 \
    wandb.project=bi-state-dr-low-1 \
    wandb.continue_run_id=p1dj22xx \
    dryrun=false
