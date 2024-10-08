#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=fph_mppo_low

python -m src.train.ppo \
    base_policy.wandb_id=fph-state-dr-low-1/v9fcbu9w \
    base_policy.wt_type=best_success_rate \
    env.task=factory_peg_hole \
    env.randomness=low \
    num_env_steps=200 \
    kl_coef=0.5 \
    gae_lambda=1.0 \
    gamma=1.0 \
    init_logstd=-4.0 \
    vf_coef=1.0 \
    wandb.project=fph-mlp-ppo-dr-low-1 \
    debug=true
