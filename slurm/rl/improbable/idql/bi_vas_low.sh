#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=bi_vas_low

python -m src.train.vas \
    base_policy.wandb_id=bi-state-dr-low-1/p1dj22xx \
    base_policy.wt_type=best_success_rate \
    env.task=bimanual_insertion \
    env.randomness=low \
    control.controller=dexhub \
    num_env_steps=500 \
    num_envs=256 \
    n_iterations_train_only_value=5 \
    eval_interval=20 \
    checkpoint_interval=100 \
    total_timesteps=1_000_000_000 \
    wandb.project=bi-rppo-dr-low-1 \
    debug=false
