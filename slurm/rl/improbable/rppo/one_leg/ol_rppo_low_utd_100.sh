#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_rppo_low_utd_100

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=ol-state-dr-1/e3d4a367 \
    base_policy.wt_type=best_success_rate \
    env.randomness=low \
    update_epochs=100 \
    actor.residual_policy.init_logstd=-0.9 \
    actor.residual_policy.learn_std=true \
    ent_coef=0.0 \
    sample_perturbations=false \
    debug=false
