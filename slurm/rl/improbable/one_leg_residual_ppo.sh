#!/bin/bash

#SBATCH -p vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ol_rppo_low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo \
    debug=false
