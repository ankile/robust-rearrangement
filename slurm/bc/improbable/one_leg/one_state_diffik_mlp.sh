#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-v100,vision-pulkitag-h100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=oneleg_state_diffik_mlp
#SBATCH --output=output_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=state/mlp_diffik_rollout task=one_leg \
    action_horizon=4 pred_horizon=4 \
    actor.hidden_dims='[1024, 1024]' \
    obs_horizon=3 actor.dropout=0.5 \
    dryrun=false