#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=00-12:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_diff_unet_sm

# Make an infinite loop that runs the rollout script

# run_id="ol-state-dr-med-1/9zjnzg4r"
run_id="residual-ppo-dr-1/7mv6o4i9"
wt_type="best_success_rate"
rollout_suffix="rppo_1"

while true; do
    python -m src.eval.evaluate_model --run-id ol-state-dr-med-1/runs/9zjnzg4r --n-envs 32 \
        --n-rollouts 32 -f one_leg --if-exists append --max-rollout-steps 750 --controller diffik \
        --use-new-env --action-type pos --observation-space image --randomness med --wt-type $wt_type \
        --save-rollouts --save-rollouts-suffix $rollout_suffix
done