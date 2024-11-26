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
#SBATCH --job-name=1a_bi_mlp_si_low

python -m src.train.bc +experiment=state/mlp_lg_ch \
    randomness=low \
    task=bimanual_insertion \
    control.controller=dexhub \
    rollout.num_envs=128 \
    rollout.max_steps=400 \
    rollout.count=512 \
    pred_horizon=1 \
    action_horizon=1 \
    wandb.name=mlp-lg-ch-20 \
    wandb.project=bi-state-dr-low-1 \
    wandb.continue_run_id=23b2fc6e \
    dryrun=false
