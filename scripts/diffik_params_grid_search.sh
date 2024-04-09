#!/bin/bash

# Define the parameter ranges
pos_scalars="1"
rot_scalars="1"
stiffnesses="500 600 700 800 900"
dampenings="50 150 250 350"

results_file="/data/scratch/ankile/results.csv"

# Create the CSV file and add the header
# NOTE: Uncomment the following line if you want to overwrite the existing results.csv file
# echo "pos_scalar,rot_scalar,stiffness,damping,success_rate" > $results_file

# Calculate the total number of parameter combinations
total_combinations=20
current_combination=1

# Iterate over the parameter ranges
for pos_scalar in $pos_scalars; do
    for rot_scalar in $rot_scalars; do
        for stiffness in $stiffnesses; do
            for damping in $dampenings; do
                echo "Current parameters:"
                echo "  pos_scalar: $pos_scalar"
                echo "  rot_scalar: $rot_scalar"
                echo "  stiffness: $stiffness"
                echo "  damping: $damping"
                echo "Progress: $current_combination/$total_combinations"
                echo "---------------------------------------"
                
                # Run the Python script with the current parameter combination
                success_rate=$(python -m src.gym.diffik_params_grid_search $pos_scalar $rot_scalar $stiffness $damping | tail -n1)
                
                # Append the results to the CSV file
                echo "$pos_scalar,$rot_scalar,$stiffness,$damping,$success_rate" >> $results_file
                
                current_combination=$((current_combination + 1))
            done
        done
    done
done