#!/bin/bash

# Array of task types
tasks=(one_leg round_table lamp mug_rack factory_peg_hole)

# Array of randomness types
randomnesses=(low med)

export DATA_DIR_PROCESSED=/data/scratch/ankile/data_processed_new

# Nested loop to iterate through all combinations
for task in "${tasks[@]}"; do
    for randomness in "${randomnesses[@]}"; do
        # Execute the command with current parameters
        python -m src.data_processing.process_pickles -c diffik -d sim -f "$task" -s teleop -r "$randomness" -o success --n-cpus 4 --max-files 50
    done
done