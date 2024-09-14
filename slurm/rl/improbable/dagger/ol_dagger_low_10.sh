#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=ol_dagger_low_10

python -m src.train.dagger \
    student_policy.wandb_id=ol-state-dr-low-1/6i7hupje \
    teacher_policy.wandb_id=ol-rppo-dr-low-1/k8tg86rc \
    env.randomness=low \
    beta=1.0 \
    correct_student_action_only=false \
    debug=false
