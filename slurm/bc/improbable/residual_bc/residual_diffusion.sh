#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ol_state_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=00-12:00
#SBATCH --gres=gpu:1

python -m src.train.bc +experiment=state/residual_diffusion \
    task=one_leg rollout=rollout \
    dryrun=false