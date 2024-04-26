#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ppo_oneleg_chunked
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/scratch/ankile/miniconda3/envs/rr/lib
# conda activate /data/scratch/ankile/miniconda3/envs/rr

# Run your command with the provided arguments
python -m src.train.ppo --num-env-steps 800 --data-collection-steps 100 --num-envs 2048 --bc-coef 0.90 \
    --learning-rate 1e-5 --save-model --total-timesteps 60000000 \
    --headless --exp-name oneleg --no-normalize-reward --update-epochs 10 \
    --no-normalize-obs --no-clip-vloss --num-minibatches 64 --init-logstd -4 \
    --agent residual-big --ee-dof 10 --bc-loss-type mse --no-supervise-value-function \
    --action-type pos --no-adaptive-bc-coef --min-bc-coef 0.90 --n-decrease-lr 0 --chunk-size 8