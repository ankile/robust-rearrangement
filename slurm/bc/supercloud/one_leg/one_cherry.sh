#!/bin/bash
#SBATCH -p xeon-g6-volta
#SBATCH -t 0-20:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH -o wandb_output_%j.log
#SBATCH -e wandb_error_%j.log

# Load any modules or set up the environment if needed
source /etc/profile
module load anaconda/2022b
module load cuda/11.3
source activate furniture-env
export LD_LIBRARY_PATH=/home/gridsan/asimeono/.conda/envs/furniture-env/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/gridsan/asimeono/.conda/envs/furniture-env/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export AWS_COMMAND='/home/gridsan/asimeono/aws-cli/v2/current/bin/aws'
export DATA_DIR_PROCESSED="/home/gridsan/asimeono/data/furniture-data/"
export WANDB_ENTITY="robot-rearrangement"
alias aws='/home/gridsan/asimeono/aws-cli/v2/current/bin/aws'
cd ~/repos/research/furniture-diffusion


python -m src.train.bc_no_rollout \
    +experiment=image_cherry_noise \
    furniture=one_leg \
    data.dataloader_workers=20 \
    wandb.mode=offline
