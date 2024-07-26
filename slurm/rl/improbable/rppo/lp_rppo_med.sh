#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=lp_rppo_med

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=lp-state-dr-med-1/fziwvs8k \
    base_policy.wt_type=best_success_rate \
    env.task=lamp \
    env.randomness=med \
    num_env_steps=1000 \
    num_envs=1024 \
    n_iterations_train_only_value=0 \
    actor.residual_policy.init_logstd=-1.0 \
    actor.residual_policy.learn_std=false \
    total_timesteps=1000000000 \
    wandb.project=lp-rppo-dr-med-1 \
    wandb.continue_run_id=ev23t35c \
    debug=false
