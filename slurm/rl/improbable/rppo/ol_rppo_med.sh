#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-a100,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ol_rppo_med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=ol-state-dr-med-1/9zjnzg4r \
    env.task=one_leg \
    env.randomness=med \
    num_env_steps=700 \
    normalize_reward=false \
    wandb.project=ol-rppo-dr-med-1 \
    debug=false