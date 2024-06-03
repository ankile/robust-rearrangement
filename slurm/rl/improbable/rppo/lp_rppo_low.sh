#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=lp_rppo_low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=lp-state-dr-low-1/hwj7kk4i \
    base_policy.wt_type=best_success_rate \
    env.task=lamp \
    env.randomness=low \
    num_env_steps=1000 \
    wandb.project=lp-rppo-dr-low-1 \
    debug=false