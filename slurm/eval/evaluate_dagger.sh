#!/bin/bash

# Output CSV file
curr_time=$(date "+%Y.%m.%d-%H.%M.%S")
output_file="dagger_eval_$curr_time.csv"

# Create CSV header
echo "id,idx,success_rate" > "$output_file"

# Array of IDs
ids=("v8zfflw4" "9ks1mzyv" "o8anvsl3" "nu86h12r" "066chnig" "n8yhgylx" "lsc0qa3l")

for id in "${ids[@]}"; do
    echo "Processing ID: $id"
    
    for idx in {1..40}; do
        echo "Processing idx: $idx"
        
        # Run the command and capture its output
        output=$(python -m src.eval.evaluate_model --n-envs 1024 --n-rollouts 1024 -f one_leg --if-exists append --max-rollout-steps 700 --controller diffik --use-new-env --action-type pos --randomness low --wt-type "_$idx.pt" --run-id "ol-dagger-low-1/$id" --observation-space state)
        
        # Extract the success rate using grep and awk
        success_rate=$(echo "$output" | grep "Success rate:" | awk '{print $3}' | sed 's/%//')
        
        # Append the result to the CSV file
        echo "$id,$idx,$success_rate" >> "$output_file"
        
        echo "Completed processing for ID: $id, idx: $idx"
        echo "------------------------"
    done
done

echo "All evaluations complete. Results written to $output_file"