#!/bin/bash

#SBATCH -p seas_gpu
#SBATCH -t 7-00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 16
#SBATCH --account=parkes_low_priority
#SBATCH --job-name=3_rt_50k_low_state

python -m src.train.bc +experiment=state/scaling/round_table/50k \
    actor.diffusion_model.down_dims='[512, 1024, 2048]' \
    dryrun=false
