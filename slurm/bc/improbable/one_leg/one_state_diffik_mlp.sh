#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=oneleg_state_diffik_mlp
#SBATCH --output=output_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.bc +experiment=state/mlp furniture=one_leg dryrun=false \
    rollout=rollout rollout.every=5 rollout.max_steps=650 rollout.num_envs=512 \
    pred_horizon=16 action_horizon=8 control.controller=diffik actor.hidden_dims='[1024,1024,1024,1024,1024]' \
    demo_source='[teleop,rollout]' data.data_subset=100