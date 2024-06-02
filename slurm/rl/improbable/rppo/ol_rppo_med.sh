#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-v100,vision-pulkitag-a100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=ol_rppo_med
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=ol-state-dr-med-1/9zjnzg4r \
    reset_every_iteration=false \
    reset_on_success=true \
    debug=false
