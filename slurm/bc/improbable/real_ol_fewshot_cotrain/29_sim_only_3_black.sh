#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=28_ol_cotrain_40_black_cf3

python -m src.train.bc +experiment=image/real_ol_cotrain \
    actor/diffusion_model=transformer \
    training.actor_lr=1e-4 \
    training.num_epochs=5000 \
    demo_source='[teleop,rollout]' \
    task='[one_leg_render_demos_brighter,one_leg_render_rppo_brighter,one_leg_render_demos_black,one_leg_render_rppo_black]' \
    randomness='[low,med,med_perturb]' \
    environment=sim \
    dryrun=false
