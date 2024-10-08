#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ol_dag_low_decay_scratch_img

python -m src.train.dagger \
    student_policy.wandb_id=ol-vision-sim-demo-scaling-low-1/6f6cab8d \
    teacher_policy.wandb_id=ol-rppo-dr-low-1/k8tg86rc \
    env.randomness=low \
    student_policy.wt_type=null \
    beta=1.0 \
    teacher_only_iters=4 \
    beta_linear_decay=0.025 \
    beta_min=0.1 \
    correct_student_action_only=false \
    num_envs=16 \
    num_epochs=100 \
    num_iterations=2000 \
    eval_first=false \
    max_steps_per_epoch=10 \
    learning_rate_student=1e-4 \
    replay_buffer_size=150000 \
    observation_type=image \
    batch_size=128 \
    num_iterations=1000 \
    debug=false

