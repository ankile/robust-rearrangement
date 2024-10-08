#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=fph_rppo_low

python -m src.train.residual_ppo \
    base_policy.wandb_id=fph-state-dr-low-1/qqn8b1lt \
    base_policy.wt_type=best_success_rate \
    env.task=factory_peg_hole \
    env.randomness=low \
    num_env_steps=200 \
    num_envs=1024 \
    n_iterations_train_only_value=0 \
    eval_interval=20 \
    checkpoint_interval=100 \
    actor.residual_policy.learn_std=false \
    total_timesteps=1_000_000_000 \
    wandb.project=fph-rppo-dr-low-1 \
    debug=false
