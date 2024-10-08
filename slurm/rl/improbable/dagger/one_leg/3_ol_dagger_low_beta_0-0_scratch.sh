#!/bin/bash

#SBATCH -p vision-pulkitag-a100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=ol_dag_low_decay_scratch_0-0

python -m src.train.dagger \
    student_policy.wandb_id=ol-state-dr-low-1/6i7hupje \
    teacher_policy.wandb_id=ol-rppo-dr-low-1/k8tg86rc \
    env.randomness=low \
    student_policy.wt_type=null \
    beta=0.9 \
    teacher_only_iters=1 \
    correct_student_action_only=false \
    num_envs=64 \
    num_epochs=50 \
    eval_first=false \
    max_steps_per_epoch=10 \
    learning_rate_student=1e-4 \
    replay_buffer_size=10000000 \
    beta_min=0.0 \
    beta_linear_decay=0.025 \
    debug=false

# #SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000
