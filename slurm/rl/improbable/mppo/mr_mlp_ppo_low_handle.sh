#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=mr_mppo_low

python -m src.train.ppo \
    base_policy.wandb_id=mr-state-dr-low-1/6zwxy8h8 \
    base_policy.wt_type=best_success_rate \
    env.task=mug_rack \
    env.randomness=low \
    num_env_steps=400 \
    num_envs=1024 \
    kl_coef=0.5 \
    gae_lambda=1.0 \
    gamma=1.0 \
    init_logstd=-4.0 \
    vf_coef=1.0 \
    wandb.project=mr-mlp-ppo-dr-low-1 \
    debug=false
