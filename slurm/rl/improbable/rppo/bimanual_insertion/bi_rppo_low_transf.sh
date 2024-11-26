#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-a100,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=bi_rppo_low

python -m src.train.residual_ppo \
    base_policy.wandb_id=bi-state-dr-low-1/p1dj22xx \
    base_policy.wt_type=best_success_rate \
    env.task=bimanual_insertion \
    env.randomness=low \
    num_env_steps=500 \
    num_envs=256 \
    n_iterations_train_only_value=0 \
    actor.residual_policy.init_logstd=-1.0 \
    actor.residual_policy.learn_std=false \
    total_timesteps=1000000000 \
    wandb.project=bi-rppo-dr-low-1 \
    wandb.continue_run_id=2dfed6ee \
    debug=false
