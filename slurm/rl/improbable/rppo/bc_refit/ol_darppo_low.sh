#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_darppo_low

python -m src.train.residual_ppo_w_bc +experiment=rl/residual_ppo_w_bc \
    base_policy.wandb_id=ol-state-dr-low-1/6i7hupje \
    base_policy.wt_type=best_success_rate \
    env.randomness=low \
    actor.residual_policy.init_logstd=-1.0 \
    actor.residual_policy.learn_std=false \
    ent_coef=0.0 \
    sample_perturbations=false \
    num_envs=128 \
    debug=true
