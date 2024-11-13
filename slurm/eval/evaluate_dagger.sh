#!/bin/bash

#SBATCH -p vision-pulkitag-a6000,vision-pulkitag-3090,vision-pulkitag-a100,vision-pulkitag-v100
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=0-12:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_dagger


# Array of IDs
# ids=("m96s3a3j")
# project="lp-dagger-low-1"
# task="lamp"
ids=("c92783d5")
project="rt-dagger-low-1"
task="round_table"
max_steps=1000

# ids=("3863e58b")
# project="fph-dagger-low-1"
# task="factory_peg_hole"
# max_steps=200


# Output CSV file

# Get the path to the folder this script is in
script_dir=$(dirname "$0")

curr_time=$(date "+%Y.%m.%d-%H.%M.%S")
output_file="$script_dir/eval_dagger_${curr_time}_$task.csv"

# Create CSV header
echo "id,idx,success_rate" > "$output_file"

for id in "${ids[@]}"; do
    echo "Processing ID: $id"
    
    for idx in {90..400..10}; do
        echo "Processing idx: $idx"
        
        # Run the command and capture its output
        output=$(python -m src.eval.evaluate_model --n-envs 1024 --n-rollouts 1024 -f $task --if-exists append --max-rollout-steps $max_steps --action-type pos --randomness low --wt-type "_$idx.pt" --run-id "$project/$id" --observation-space state)
        
        # Extract the success rate using grep and awk
        success_rate=$(echo "$output" | grep "Success rate:" | awk '{print $3}' | sed 's/%//')
        
        # Append the result to the CSV file
        echo "$id,$idx,$success_rate" >> "$output_file"
        
        echo "Completed processing for ID: $id, idx: $idx"
        echo "------------------------"
    done
done

echo "All evaluations complete. Results written to $output_file"