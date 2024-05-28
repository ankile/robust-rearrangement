#!/bin/bash

#SBATCH -p vision-pulkitag-a100,vision-pulkitag-a6000,vision-pulkitag-3090
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=eval_ol_med_best_loss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=0-04:00
#SBATCH --gres=gpu:1

python -m src.eval.evaluate_model --project-id robust-rearrangement/ol-state-dr-med-1 --n-envs 128 --n
-rollouts 128 -f one_leg --if-exists append --max-rollout-steps 700 --controller diffik --use-new-env --action-type pos --observation-space stat
e --randomness med --wt-type best_test_loss --wandb