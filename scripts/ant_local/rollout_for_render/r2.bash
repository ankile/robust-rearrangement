#!/bin/bash

# Define the shell scripts to be run
scripts=("10_rt_bc_vs_rppo_pert.sh" "11_lp_bc_vs_rppo_pert.sh" "9_ol_bc_vs_rppo_pert.sh")

# Loop 100 times
for ((i=1; i<=100; i++))
do
  echo "Iteration $i"
  for script in "${scripts[@]}"
  do
    echo "Running $script"
    sh "$script"
  done
done

echo "All iterations completed."
