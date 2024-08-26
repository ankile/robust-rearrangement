#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=lp_rppo_med

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=lp-state-dr-med-1/en9wdmzr \
    base_policy.wt_type=best_success_rate \
    env.task=lamp \
    env.randomness=med \
    num_env_steps=1000 \
    wandb.project=lp-rppo-dr-med-1 \
    debug=false
