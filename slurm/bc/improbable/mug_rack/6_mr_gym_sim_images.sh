#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256GB
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=1_ol_sim_only

python -m src.train.bc +experiment=image/diff_transformer \
    training.actor_lr=1e-4 \
    training.encoder_lr=1e-5 \
    training.num_epochs=5000 \
    demo_source='[teleop,rollout]' \
    task=mug_rack \
    randomness=low \
    data.data_paths_override='[diffik/sim/mug_rack_gym_rerender/rollout/low/success/mug_rack_rerender_0.zarr,diffik/sim/mug_rack_gym_rerender/rollout/low/success/mug_rack_rerender_250.zarr]' \
    actor.confusion_loss_beta=0.0 \
    environment=sim \
    wandb.project=mr-image-1 \
    dryrun=false

    # data.data_paths_override='[diffik/sim/mug_rack_gym_rerender/rollout/low/success/mug_rack_rerender_0.zarr,diffik/sim/mug_rack_gym_rerender/rollout/low/success/mug_rack_rerender_250.zarr,diffik/sim/mug_rack_gym_rerender/rollout/low/success/mug_rack_rerender_500.zarr,diffik/sim/mug_rack_gym_rerender/rollout/low/success/mug_rack_rerender_750.zarr]' \
    # +data.max_episode_count.mug_rack_gym_rerender.rollout.low.success=500 \
