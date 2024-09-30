#!/bin/bash

#SBATCH -p vision-pulkitag-v100,vision-pulkitag-a100,vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_rppo_low_alpha_02

# ln(exp(-1.0) * 0.1 / 0.05) = -0.3

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=ol-state-dr-low-1/6i7hupje \
    base_policy.wt_type=best_success_rate \
    env.randomness=low \
    actor.residual_policy.init_logstd=-0.3 \
    actor.residual_policy.learn_std=false \
    actor.residual_policy.action_scale=0.05 \
    sample_perturbations=false \
    debug=false
