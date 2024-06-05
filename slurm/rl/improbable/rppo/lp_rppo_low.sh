#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=lp_rppo_low_roll

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=lp-state-dr-low-1/b5dcl1tt \
    base_policy.wt_type=best_success_rate \
    env.task=lamp \
    env.randomness=low \
    num_env_steps=1000 \
    num_envs=1024 \
    wandb.project=lp-rppo-dr-low-1 \
    debug=false
