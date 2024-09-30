#!/bin/bash

#SBATCH -p vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_mlp_rppo_med

python -m src.train.residual_ppo +experiment=rl/residual_mlp_ppo \
    base_policy.wandb_id=ol-state-dr-med-1/844swkwn \
    base_policy.wt_type=best_success_rate \
    env.randomness=med \
    actor.residual_policy.init_logstd=-1.0 \
    actor.residual_policy.learn_std=false \
    ent_coef=0.0 \
    sample_perturbations=false \
    debug=false
