#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --time=01-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=uncut_videos

# Make an infinite loop that runs the rollout script

# === One Leg ===

# Low BC run
task="one_leg"
run_id="ol-state-dr-1/e3d4a367"
rollout_suffix="bc"
randomness="low"
rollout_steps=700

# Low RL run
# run_id="ol-rppo-dr-low-1/k8tg86rc"
# rollout_suffix="rppo"
# randomness="low"
# rollout_steps=700
# task="one_leg"

wt_type="best_success_rate"


# === Round Table ===

# Med BC run
# run_id="rt-state-dr-med-1/pb4urpt5"
# rollout_suffix="bc"
# wt_type="best_success_rate"

# Med RL run
# run_id="rt-rppo-dr-med-1/k737s8lj"
# rollout_suffix="rppo"
# wt_type="latest"

# randomness="med"
# task="round_table"
# rollout_steps=1000

while true; do
    DATA_DIR_RAW=/data/scratch/ankile/robust-assembly-video-data python -m src.eval.evaluate_model \
        --n-envs 32 --n-rollouts 32 -f $task --if-exists append --max-rollout-steps $rollout_steps \
        --action-type pos --randomness $randomness --wt-type $wt_type --run-id $run_id \
        --observation-space image --save-rollouts --save-failures --save-rollouts-suffix $rollout_suffix \
        --compress-pickles
done