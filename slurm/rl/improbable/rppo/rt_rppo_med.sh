#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=rt_rppo_med

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=rt-state-dr-med-1/pb4urpt5 \
    base_policy.wt_type=best_success_rate \
    env.task=round_table \
    env.randomness=med \
    num_env_steps=1000 \
    wandb.project=rt-rppo-dr-med-1 \
    debug=false
