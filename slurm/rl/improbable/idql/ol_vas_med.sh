#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_vas_low
#SBATCH --requeue

python -m src.train.vas \
    base_policy.wandb_id=ol-state-dr-med-1/9zjnzg4r \
    base_policy.wt_type=best_success_rate \
    env.task=one_leg \
    env.randomness=med \
    num_env_steps=700 \
    num_envs=1024 \
    n_iterations_train_only_value=0 \
    eval_interval=20 \
    checkpoint_interval=100 \
    update_epochs=5 \
    sigma=0.0 \
    eta=1.0 \
    anneal_lr=true \
    wandb.project=ol-vas-dr-med-1 \
    debug=false
