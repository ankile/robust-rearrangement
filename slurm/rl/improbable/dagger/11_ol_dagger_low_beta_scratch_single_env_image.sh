#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=11_ol_dagger_low_beta_scratch_single_env_image

python -m src.train.dagger \
    student_policy.wandb_id=ol-vision-sim-demo-scaling-low-1/6f6cab8d \
    teacher_policy.wandb_id=ol-rppo-dr-low-1/k8tg86rc \
    env.randomness=low \
    student_policy.wt_type=best_success_rate \
    beta=1.0 \
    teacher_only_iters=10 \
    correct_student_action_only=false \
    eval_interval=10 \
    num_envs=1 \
    num_epochs=5 \
    eval_first=false \
    beta_min=0.5 \
    beta_decay_ref_sr_ratio=0.8 \
    observation_type=image \
    beta_linear_decay=0.01 \
    max_steps_per_epoch=10 \
    checkpoint_interval=10 \
    learning_rate_student=1e-4 \
    replay_buffer_size=100000 \
    batch_size=256 \
    num_iterations=1000 \
    debug=false
