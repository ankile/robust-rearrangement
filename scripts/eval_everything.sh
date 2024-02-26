#!/bin/bash

# Define associative array for project and furniture pairs
declare -A projects
projects=(
    ["one_leg-baseline-1"]="one_leg"
    ["one_leg-trajaug-1"]="one_leg"
    ["one_leg-collectinfer-1"]="one_leg"
    ["one_leg-trajaug_infer-1"]="one_leg"
    ["one_leg-cherry-1"]="one_leg"
    ["round_table-baseline-1"]="round_table"
    ["round_table-trajaug-1"]="round_table"
    ["round_table-collectinfer-1"]="round_table"
    ["round_table-trajaug_infer-1"]="round_table"
    ["round_table-cherry-1"]="round_table"
    ["lamp-baseline-1"]="lamp"
    ["lamp-trajaug-1"]="lamp"
    ["lamp-collectinfer-1"]="lamp"
    ["lamp-trajaug_infer-1"]="lamp"
    ["lamp-cherry-1"]="lamp"
    ["square_table-baseline-1"]="square_table"
    ["square_table-trajaug-1"]="square_table"
    ["square_table-collectinfer-1"]="square_table"
    ["square_table-trajaug_infer-1"]="square_table"
    ["square_table-cherry-1"]="square_table"
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

     # Construct and execute the command
    command="python -m src.eval.evaluate_model --n-envs 10 --n-rollouts 10 --randomness low -f round_table --sweep-id sweeps/sweeps/sw6ryrnf --wandb --if-exists append --action-type pos --prioritize-fewest-rollout --max-rollouts 100"
    echo "Executing: $command"
    
    # Execute the command; redirect output to log file and stderr to a separate file
    # $command >> "round_table_sweep_output.log" 2>> "round_table_sweep_errors.log"
    eval $command
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Execution for round_table_sweep completed successfully."
    else
        echo "Error encountered during execution for round_table_sweep. Check round_table_sweep_errors.log for details."
    fi

     # Construct and execute the command
    command="python -m src.eval.evaluate_model --n-envs 10 --n-rollouts 10 --randomness low -f square_table --sweep-id sweeps/sweeps/44hsqeuy --wandb --if-exists append --action-type pos --prioritize-fewest-rollout --max-rollouts 100"
    echo "Executing: $command"
    
    # Execute the command; redirect output to log file and stderr to a separate file
    # $command >> "square_table_sweep_output.log" 2>> "square_table_sweep_errors.log"
    eval $command
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Execution for square_table_sweep completed successfully."
    else
        echo "Error encountered during execution for square_table_sweep. Check square_table_sweep_errors.log for details."
    fi

done
