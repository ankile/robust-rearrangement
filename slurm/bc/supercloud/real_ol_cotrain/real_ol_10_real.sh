#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:volta:1
#SBATCH --job-name=real_ol_10_real
#SBATCH -c 20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=5e-5 \
    training.num_epochs=2000 \
    actor.confusion_loss_beta=0.0 \
    demo_source=teleop \
    furniture=one_leg_simple \
    randomness=low \
    environment=real \
    +data.max_episode_count.one_leg_simple.teleop.low.success=10 \
    wandb.mode=offline \
    dryrun=false

