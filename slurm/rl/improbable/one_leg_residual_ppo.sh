#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=oneleg_residual_ppo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

git checkout residual-ppo

python -m src.train.residual_ppo --num-env-steps 800 --data-collection-steps 800 --num-envs 1536 --bc-coef 0.0 \
    --learning-rate 3e-4 --save-model --total-timesteps 200000000 --headless --exp-name oneleg \
    --normalize-reward --update-epochs 2 \
    --no-normalize-obs --no-clip-vloss --num-minibatches 4 --init-logstd -4 \
    --ee-dof 10 --agent bigger-residual --target-kl 0.1 \
    --action-type pos --gamma 0.997 --n-iterations-train-only-value 5 --residual_regularization 0.1 \
    --no-debug