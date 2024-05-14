#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ol_res_ppo_new
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo \
    learning_rate=3e-4 \
    residual_regularization=0.01 \
    n_iterations_train_only_value=5 \
    residual_policy.init_logstd=-4 \
    residual_policy.action_head_std=0.01 \
    normalize_reward=true \
    num_minibatches=2 \
    update_epochs=4 \
    gamma=0.998 \
    gae_lambda=0.95 \
    num_envs=1024 \
    num_env_steps=850 \
    target_kl=0.03 \
    debug=false
