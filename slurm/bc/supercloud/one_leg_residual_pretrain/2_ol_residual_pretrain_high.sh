#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_res_pre_high
#SBATCH -c 20

python -m src.train.bc +experiment=state/residual_diffusion \
    actor/diffusion_model=unet \
    training.actor_lr=1e-4 \
    training.num_epochs=5000 \
    training.batch_size=256 \
    demo_source=teleop \
    task='[one_leg]' \
    rollout.task=one_leg \
    randomness='[high,high_perturb]' \
    environment='[sim]' \
    wandb.mode=offline \
    wandb.continue_run_id=nh13mzy4 \
    dryrun=false
