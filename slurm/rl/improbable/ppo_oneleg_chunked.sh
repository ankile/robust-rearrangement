#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ppo_oneleg_chunked
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-04:00
#SBATCH --gres=gpu:1

# Run your command with the provided arguments
python -m src.train.ppo --num-env-steps 1024 --data-collection-steps 128 --num-envs 2048 \
    --bc-coef 0.90 --learning-rate 1e-4 --save-model --total-timesteps 60000000 --headless \
    --exp-name oneleg --normalize-reward --update-epochs 5 --bc-loss-type nll \
    --no-normalize-obs --no-clip-vloss --num-minibatches 4 --init-logstd -3 \
    --agent residual-separate --ee-dof 10 --clip-coef 0.02 \
    --action-type pos --chunk-size 8 --gamma 0.95 --n-iterations-train-only-value 0