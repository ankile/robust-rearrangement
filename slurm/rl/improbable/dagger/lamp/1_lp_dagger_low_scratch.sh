#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=1_lp_dagger_low_scratch

python -m src.train.dagger \
    student_policy.wandb_id=lp-state-dr-low-1/b5dcl1tt \
    teacher_policy.wandb_id=lp-rppo-dr-low-1/hd2i5gje \
    env.randomness=low env.task=lamp \
    num_env_steps=800 \
    student_policy.wt_type=null \
    beta=0.95 \
    teacher_only_iters=2 \
    correct_student_action_only=false \
    eval_interval=5 \
    num_envs=16 \
    num_epochs=100 \
    eval_first=false \
    num_iterations=500 \
    beta_min=0.5 \
    beta_decay_ref_sr_ratio=0.8 \
    beta_linear_decay=0.1 \
    max_steps_per_epoch=100 \
    checkpoint_interval=1 \
    learning_rate_student=1e-4 \
    replay_buffer_size=10000000 \
    wandb.project=lp-dagger-low-1 \
    wandb.continue_run_id=null \
    debug=false
