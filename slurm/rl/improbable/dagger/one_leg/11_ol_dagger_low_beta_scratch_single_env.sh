#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=ol_dag_low_decay_scratch

python -m src.train.dagger \
    student_policy.wandb_id=ol-state-dr-low-1/6i7hupje \
    teacher_policy.wandb_id=ol-rppo-dr-low-1/k8tg86rc \
    env.randomness=low \
    student_policy.wt_type=null \
    beta=1.0 \
    teacher_only_iters=10 \
    correct_student_action_only=false \
    eval_interval=10 \
    num_envs=1 \
    num_epochs=5 \
    eval_first=false \
    beta_min=0.5 \
    beta_decay_ref_sr_ratio=0.8 \
    beta_linear_decay=0.01 \
    max_steps_per_epoch=10 \
    checkpoint_interval=10 \
    learning_rate_student=1e-4 \
    replay_buffer_size=10000000 \
    num_iterations=1000 \
    debug=false
