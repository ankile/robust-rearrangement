#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=rt_rppo_med

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=rt-state-dr-med-1/n5g6x9jg \
    base_policy.wt_type=best_success_rate \
    env.task=round_table \
    env.randomness=med \
    num_env_steps=1000 \
    actor.residual_policy.init_logstd=-1.5 \
    actor.residual_policy.learn_std=true \
    total_timesteps=1000000000 \
    ent_coef=0.001 \
    wandb.project=rt-rppo-dr-med-1 \
    debug=false
