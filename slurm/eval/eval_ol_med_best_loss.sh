#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=eval_ol_med_best_loss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=0-04:00
#SBATCH --gres=gpu:1

project_id="ol-state-dr-low-1"
n_rollouts=256
randomness="low"

python -m src.eval.evaluate_model --project-id robust-rearrangement/$project_id --n-envs $n_rollouts \
    --n-rollouts $n_rollouts -f one_leg --if-exists overwrite --max-rollout-steps 750 --controller diffik \
    --use-new-env --action-type pos --observation-space state --randomness $randomness --wt-type best_test_loss --wandb