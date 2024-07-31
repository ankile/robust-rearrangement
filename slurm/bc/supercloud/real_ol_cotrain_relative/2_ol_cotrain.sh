#!/bin/bash

#SBATCH -p xeon-g6-volta
#SBATCH -t 3-00:00
#SBATCH --gres=gpu:volta:4
#SBATCH -c 80
#SBATCH --job-name=1_ol_50_gpu

OMP_NUM_THREADS=20

python -m src.train.bc +experiment=image/real_ol_cotrain \
    demo_source='[teleop,rollout]' \
    furniture='[one_leg_render_rppo,one_leg_full,one_leg_highres]' \
    randomness='[low,med,med_perturb]' \
    environment='[real,sim]' \
    training.batch_size=64 \
    data.dataloader_workers=20 \
    control.control_mode=relative \
    training.clip_grad_norm=true \
    wandb.mode=offline \
    wandb.project=real-ol-cotrain-relative-1 \
    dryrun=false
