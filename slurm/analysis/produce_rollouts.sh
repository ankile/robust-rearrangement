#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=00-02:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=make_traj_bc_med

# Make an infinite loop that runs the rollout script

# Low BC run
# run_id="ol-state-dr-low-1/6i7hupje"
# rollout_suffix="bc_unet"
# randomness="low"

# Low RL run
# run_id="residual-ppo-dr-low-1/kzlx4y3f"
# rollout_suffix="rppo_low" 
# randomness="low"

# Med BC run
run_id="ol-state-dr-med-1/9zjnzg4r"
rollout_suffix="bc_unet"
randomness="med"

# Med RL run
# run_id="residual-ppo-dr-med-1/6xmgqdiw"
# run_id="residual-ppo-dr-med-1/h7dg0og4"
# rollout_suffix="rppo_med"
# randomness="med"

wt_type="best_success_rate"

while true; do
    python -m src.eval.evaluate_model --run-id $run_id --n-envs 32 \
        --n-rollouts 32 -f one_leg --if-exists append --max-rollout-steps 750 --controller diffik \
        --use-new-env --action-type pos --observation-space image --randomness $randomness --wt-type $wt_type \
        --save-rollouts --save-rollouts-suffix $rollout_suffix
done