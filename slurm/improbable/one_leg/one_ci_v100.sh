#!/bin/bash

#SBATCH -p vision-pulkitag-v100
#SBATCH -q vision-pulkitag-main
#SBATCH -t 1-08:00
#SBATCH --mem=64G            
#SBATCH --gres=gpu:1          
#SBATCH -c 10
#SBATCH -o wandb_output_%j.log  
#SBATCH -e wandb_error_%j.log

export WANDB_CACHE_DIR="/data/scratch/ankile/.cache"
export WANDB_DATA_DIR="/data/scratch/ankile/.cache"
export WANDB_ARTIFACT_DIR="/data/scratch/ankile/.cache"
export WANDB_DIR="/data/scratch/ankile/.cache"
export TRANSFORMERS_CACHE="/data/scratch/ankile/.cache"
export PIP_CACHE_DIR="/data/scratch/ankile/.cache"

export TORCH_HOME="/data/scratch/ankile/.cache"
export TORCH_EXTENSIONS_DIR="/data/scratch/ankile/.cache"
export HF_HOME=/data/scratch/ankile/.cache/huggingface

export DATA_DIR_PROCESSED="/data/scratch/ankile/furniture-data/"
export DATA_DIR_RAW="/data/scratch-oc40/pulkitag/ankile/furniture-data/"

export PATH=/afs/csail.mit.edu/u/a/ankile/aws-cli/v2/current/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/scratch/ankile/miniconda3/envs/rlgpu/lib

source /data/scratch/ankile/.config/.wandb_key

python -m src.train.bc_no_rollout +experiment=image_collect_infer furniture=one_leg data.dataloader_workers=10 training.batch_size=128
