#!/bin/bash

# Define associative array for project and furniture pairs
declare -A projects
projects=(
    # one_leg
    ["one_leg-baseline-1"]="one_leg"
    ["one_leg-trajaug-1"]="one_leg"
    ["one_leg-trajaug-ratio-1"]="one_leg"
    ["one_leg-collectinfer-1"]="one_leg"
    ["one_leg-trajaug_infer-1"]="one_leg"
    ["one_leg-cherry-1"]="one_leg"
    
    # round_table
    ["round_table-baseline-1"]="round_table"
    ["round_table-trajaug-1"]="round_table"
    ["round_table-collectinfer-1"]="round_table"
    ["round_table-trajaug_infer-1"]="round_table"
    ["round_table-cherry-1"]="round_table"
    
    # lamp
    ["lamp-baseline-1"]="lamp"
    ["lamp-trajaug-1"]="lamp"
    ["lamp-collectinfer-1"]="lamp"
    ["lamp-trajaug_infer-1"]="lamp"
    ["lamp-cherry-1"]="lamp"
    
    # square_table
    ["square_table-baseline-1"]="square_table"
    ["square_table-trajaug-1"]="square_table"
    ["square_table-collectinfer-1"]="square_table"
    ["square_table-trajaug_infer-1"]="square_table"
    ["square_table-cherry-1"]="square_table"
    
    # # Curriculum
    # ["round_table-curriculum-1"]="round_table"

    # MLP
    ["one_leg-mlp-10M-baseline-1"]="one_leg"
    ["round_table-mlp-10M-baseline-1"]="round_table"
    ["lamp-mlp-10M-baseline-1"]="lamp"
    ["square_table-mlp-10M-baseline-1"]="square_table"

    # Scaling analysis
    ["one_leg-data-scaling-1"]="one_leg"
    ["one_leg-bootstrap-1"]="one_leg"
    
) 

# Define an associative array for sweeps and furniture pairs

# Infinite loop
while true; do
    # Loop through each project-furniture pair
    for project_id in "${!projects[@]}"; do
        furniture="${projects[$project_id]}"

        # Construct and execute the command
        command="python -m src.eval.evaluate_model --n-envs 10 --n-rollouts 10 --randomness low -f $furniture --project-id $project_id --wandb --if-exists append --action-type pos --run-state finished --prioritize-fewest-rollout  --max-rollouts 100"
        echo "Executing: $command"
        
        # Execute the command; redirect output to log file and stderr to a separate file
        # $command >> "${project_id}_output.log" 2>> "${project_id}_errors.log"
        eval $command
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Execution for $project_id completed successfully."
        else
            echo "Error encountered during execution for $project_id. Check ${project_id}_errors.log for details."
        fi
    done

done
