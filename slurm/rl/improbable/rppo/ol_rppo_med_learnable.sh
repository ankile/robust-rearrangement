#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_rppo_med_learnable

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=ol-state-dr-med-1/9zjnzg4r \
    env.task=one_leg \
    env.randomness=med \
    num_env_steps=700 \
    normalize_reward=false \
    wandb.project=ol-rppo-dr-med-1 \
    actor.residual_policy.init_logstd=-0.9 \
    actor.residual_policy.learn_std=true \
    ent_coef=0.001 \
    total_timesteps=1000000000 \
    sample_perturbations=false \
    wandb.continue_run_id=oipdyimz \
    debug=false
