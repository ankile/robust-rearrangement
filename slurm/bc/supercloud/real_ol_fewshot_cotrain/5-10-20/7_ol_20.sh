#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 2-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=1_ol_20
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=1e-4 \
    training.num_epochs=5000 \
    demo_source=teleop \
    task='[one_leg_full]' \
    randomness=low \
    environment=real \
    +data.max_episode_count.one_leg_full.teleop.low.success=20 \
    wandb.mode=offline \
    dryrun=false

