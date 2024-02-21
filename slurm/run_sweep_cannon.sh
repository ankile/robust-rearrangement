#!/bin/bash

#SBATCH -p gpu                # Specify the partition or queue
#SBATCH -t 0-15:00            # Set the time limit to 15 hours
#SBATCH --mem=128G            # Request 128GB of memory
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH -c 32                 # Request 32 CPUs
#SBATCH -o wandb_output_%j.log  # Output file
#SBATCH -e wandb_error_%j.log   # Error file

# Load any modules or set up the environment if needed

# Run the wandb agent command
wandb agent robot-rearrangement/sweeps/44hsqeuy
