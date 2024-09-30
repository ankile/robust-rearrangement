#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=real_ol_res_pre_high

python -m src.train.bc +experiment=state/residual_diffusion \
    training.actor_lr=1e-4 \
    training.num_epochs=5000 \
    training.batch_size=256 \
    demo_source=teleop \
    task='[one_leg]' \
    rollout.task=one_leg \
    randomness='[high,high_perturb]' \
    environment='[sim]' \
    wandb.mode=online \
    wandb.continue_run_id=nh13mzy4 \
    dryrun=false
