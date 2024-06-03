#!/bin/bash

#SBATCH -p vision-pulkitag-3090,vision-pulkitag-a6000,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --job-name=ol_rppo_high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1

python -m src.train.residual_ppo +experiment=rl/residual_ppo \
    base_policy.wandb_id=ol-state-dr-high-1/jukzzw0p \
    base_policy.wt_type=best_success_rate \
    actor.residual_policy.pretrained_wts=/data/scratch/ankile/robust-rearrangement/models/1717361845__residual_ppo__ResidualPolicy__3555185695/actor_chkpt_best_success_rate.pt \
    wandb.continue_run_id=lipfcnq2 \
    env.randomness=high \
    debug=false
