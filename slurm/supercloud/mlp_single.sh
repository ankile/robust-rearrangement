#!/bin/bash

#SBATCH -p xeon-g6-volta                # Specify the partition or queue
#SBATCH -t 0-15:00            # Set the time limit to 15 hours
#SBATCH --gres=gpu:volta:1          # Request 1 GPU
#SBATCH -c 20                 # Request 32 CPUs
#SBATCH -o wandb_output_%j.log  # Output file
#SBATCH -e wandb_error_%j.log   # Error file

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

# Run - modified prediction and action horizon
# python -m src.train.bc_no_rollout +experiment=image_mlp_10m wandb.mode=offline furniture=one_leg data.dataloader_workers=20 pred_horizon=1 action_horizon=1
python -m src.train.bc_no_rollout +experiment=image_mlp_10m wandb.mode=offline furniture=round_table data.dataloader_workers=20 pred_horizon=1 action_horizon=1
