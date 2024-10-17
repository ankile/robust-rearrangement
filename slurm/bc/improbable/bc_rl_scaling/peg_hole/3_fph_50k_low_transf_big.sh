#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=3_fph_50k_low_transf_big

python -m src.train.bc +experiment=state/scaling/peg_hole/50k \
    actor/diffusion_model=transformer_big \
    wandb.name=fph-50k-transf-big-h16-35 \
    wandb.continue_run_id=null \
    dryrun=false
