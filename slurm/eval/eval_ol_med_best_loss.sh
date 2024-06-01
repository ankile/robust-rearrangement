#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=eval_ol_high_best_success_rate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=0-02:00
#SBATCH --gres=gpu:1

project_id="ol-state-dr-high-1"
n_rollouts=256
randomness="high"
wt_type="best_success_rate" # best_test_loss, best_success_rate

python -m src.eval.evaluate_model --project-id $project_id --n-envs $n_rollouts \
    --n-rollouts $n_rollouts -f one_leg --if-exists append --max-rollout-steps 750 --controller diffik \
    --use-new-env --action-type pos --observation-space state --randomness $randomness --wt-type $wt_type --wandb