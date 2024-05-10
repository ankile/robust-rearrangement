#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ppo_oneleg_chunked
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1

git checkout residual-ppo

python -m src.train.residual_ppo --num-env-steps 1024 --data-collection-steps 1024 --num-envs 1024 --bc-coef 0.0 \
    --learning-rate 1e-5 --save-model --total-timesteps 60000000 --headless --exp-name oneleg \
    --normalize-reward --update-epochs 4 \
    --no-normalize-obs --no-clip-vloss --num-minibatches 1 --init-logstd -4 \
    --ee-dof 10 --agent residual \
    --action-type pos --gamma 0.99 --n-iterations-train-only-value 5 --residual_regularization 0.1 \
    --no-debug