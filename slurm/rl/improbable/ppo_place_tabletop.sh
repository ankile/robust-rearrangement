#!/bin/bash

#SBATCH -p vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ppo_place_tabletop
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

conda activate /data/scratch/ankile/miniconda3/envs/rr

# Run your command with the provided arguments
python -m src.train.ppo --num-env-steps 100 --num-envs 2048 --bc-coef 1 \
    --learning-rate 1e-4 --save-model --total-timesteps 40000000 \
    --headless --exp-name place-tabletop --no-normalize-reward --update-epochs 5 \
    --no-normalize-obs --no-clip-vloss --num-minibatches 64 --init-logstd -2 \
    --agent residual-big --ee-dof 10 --bc-loss-type nll \
    --no-supervise-value-function --action-type pos --adaptive-bc-coef --target-kl 0.01 \
    --min-bc-coef 0.3 --n-decrease-lr 1 --randomness-curriculum --minimum-success-rate 0.7