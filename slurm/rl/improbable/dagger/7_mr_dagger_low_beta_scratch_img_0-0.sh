#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=mr_dag_low_decay_scratch_img

python -m src.train.dagger \
    student_policy.wandb_id=ol-vision-sim-demo-scaling-low-1/6f6cab8d \
    teacher_policy.wandb_id=mr-rppo-dr-low-1/dvw6zk8e \
    env.randomness=low \
    env.task=mug_rack \
    num_env_steps=400 \
    student_policy.wt_type=null \
    beta=0.9 \
    teacher_only_iters=3 \
    correct_student_action_only=false \
    num_envs=16 \
    num_epochs=50 \
    eval_first=true \
    max_steps_per_epoch=10 \
    learning_rate_student=1e-4 \
    replay_buffer_size=150000 \
    observation_type=image \
    batch_size=128 \
    num_iterations=1000 \
    beta_min=0.0 \
    beta_linear_decay=0.025 \
    debug=false


    # student_policy.wandb_id=ol-vision-scaling-low-1/vnxxrqjx \

# #SBATCH -p vision-pulkitag-a100,vision-pulkitag-v100,vision-pulkitag-3090,vision-pulkitag-a6000
# #SBATCH -p vision-pulkitag-3090
