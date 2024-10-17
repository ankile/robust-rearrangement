#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=3_fph_50k_low_transf
#SBATCH --requeue

python -m src.train.bc +experiment=state/scaling/peg_hole/50k \
    actor/diffusion_model=transformer \
    wandb.name=fph-50k-transf-h16-34 \
    wandb.continue_run_id=807bc53a \
    dryrun=false
