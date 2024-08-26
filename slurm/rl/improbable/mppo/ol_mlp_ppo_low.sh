#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_mppo_low

python -m src.train.ppo \
    base_policy.wandb_id=ol-state-dr-low-1/173hhnou \
    base_policy.wt_type=best_success_rate \
    env.task=one_leg \
    env.randomness=low \
    num_env_steps=700 \
    num_envs=1024 \
    wandb.project=ol-rl-low-1 \
    debug=false
