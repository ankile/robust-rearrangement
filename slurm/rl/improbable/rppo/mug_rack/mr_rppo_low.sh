#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=mr_rppo_low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=mr-state-dr-low-1/qh11vha4 \
    base_policy.wt_type=best_test_loss \
    env.task=mug_rack \
    env.randomness=low\
    num_env_steps=400 \
    num_envs=1024 \
    eval_interval=20 \
    checkpoint_interval=100 \
    actor.residual_policy.init_logstd=-0.1 \
    actor.residual_policy.learn_std=false \
    total_timesteps=1000000000 \
    wandb.project=mr-rppo-dr-low-1 \
    debug=true
