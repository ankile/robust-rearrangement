#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=ol_res_ppo_new
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo \
    learning_rate=1e-4 \
    residual_regularization=0.1 \
    n_iterations_train_only_value=2 \
    residual_policy.init_logstd=-3.5 \
    residual_policy.action_head_std=0.1 \
    gamma=0.997 \
    num_envs=512 \
    debug=false
