#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=ol_rppo_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=ol-state-dr-high-1/c24b6odm \
    base_policy.wt_type=best_success_rate \
    env.randomness=high \
    debug=true
