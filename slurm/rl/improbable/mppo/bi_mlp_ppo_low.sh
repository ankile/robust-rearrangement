#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=bi_mppo_low

python -m src.train.ppo \
    base_policy.wandb_id=bi-state-dr-low-1/xi91qtpm \
    base_policy.wt_type=best_success_rate \
    env.task=bimanual_insertion \
    env.randomness=low \
    control.controller=dexhub \
    num_env_steps=500 \
    num_envs=256 \
    kl_coef=0.5 \
    init_logstd=-4.0 \
    vf_coef=1.0 \
    learning_rate=5e-5 \
    wandb.project=bi-rppo-dr-low-1 \
    debug=false
