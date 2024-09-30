#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=7_mr_sim_only

python -m src.train.bc +experiment=image/diff_transformer \
    training.actor_lr=1e-4 \
    training.encoder_lr=1e-5 \
    training.num_epochs=5000 \
    demo_source=teleop \
    task=mug_rack \
    randomness=low \
    actor.confusion_loss_beta=0.0 \
    environment=sim \
    wandb.project=mr-image-1 \
    wandb.name=diff-trans-1-demo-baseline \
    wandb.continue_run_id=pykib8wo \
    dryrun=false
