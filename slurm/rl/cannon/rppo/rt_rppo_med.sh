#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=rt_rppo_med

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=rt-state-dr-med-1/pb4urpt5 \
    base_policy.wt_type=best_success_rate \
    env.task=round_table \
    env.randomness=med \
    num_env_steps=1000 \
    wandb.project=rt-rppo-dr-med-1 \
    debug=false
