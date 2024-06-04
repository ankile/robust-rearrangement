#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=01-00:00
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --job-name=6_2_unet_lg_med_rollout

# wandb.continue_run_id=cz21iq59 \

python -m src.train.bc +experiment=state/diff_unet \
    randomness='[med,med_perturb]' \
    demo_source='[teleop,rollout]' \
    rollout.randomness=med \
    furniture=lamp \
    rollout.max_steps=1000 \
    wandb.project=lp-state-dr-med-1 \
    dryrun=false