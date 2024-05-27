#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=4_diff_tran_lg
#SBATCH -c 20

python -m src.train.bc +experiment=state/diff_tran \
    wandb.mode=offline \
    actor/diffusion_model=transformer_big \
    dryrun=false