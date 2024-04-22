#!/bin/bash

#SBATCH -p vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ppo_oneleg
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# conda activate /data/scratch/ankile/miniconda3/envs/rr

# Run your command with the provided arguments
python -m src.train.ppo --num-env-steps 800 --num-envs 2048 --bc-coef 0.9 \
    --learning-rate 1e-4 --save-model --total-timesteps 60000000 \
    --headless --exp-name oneleg --no-normalize-reward --update-epochs 5 \
    --no-normalize-obs --no-clip-vloss --num-minibatches 64 --init-logstd -3 \
    --agent residual --ee-dof 10 --bc-loss-type mse \
    --supervise-value-function --action-type pos --adaptive-bc-coef --target-kl 0.01 \
    --min-bc-coef 0.5 --n-decrease-lr 1