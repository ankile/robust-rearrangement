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
#SBATCH --job-name=ol_dag_low_decay

python -m src.train.dagger \
    student_policy.wandb_id=ol-state-dr-low-1/6i7hupje \
    teacher_policy.wandb_id=ol-rppo-dr-low-1/k8tg86rc \
    env.randomness=low \
    student_policy.wt_type=best_success_rate \
    beta=0.9 \
    teacher_only_iters=2 \
    correct_student_action_only=false \
    eval_interval=5 \
    num_envs=16 \
    num_epochs=100 \
    eval_first=false \
    beta_min=0.1 \
    max_steps_per_epoch=10 \
    checkpoint_interval=1 \
    learning_rate_student=1e-4 \
    replay_buffer_size=10000000 \
    debug=false
