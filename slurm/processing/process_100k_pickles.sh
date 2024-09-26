#!/bin/bash

#SBATCH -p vision-pulkitag-v100,vision-pulkitag-2080
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=500GB
#SBATCH --time=01-00:00
#SBATCH --job-name=process_100k_pickles

TOTAL_FILES=100000
BATCH_SIZE=1000
START=37
END=89

for i in $(seq $START $END)
do
    OFFSET=$((i * BATCH_SIZE))
    echo "Processing batch $i, Offset: $OFFSET"
    python -m src.data_processing.process_pickles -c diffik -d sim -f round_table -s rollout -r low -o success --n-cpus 4 --max-files $BATCH_SIZE --offset $OFFSET --suffix rppo --output-suffix rppo_$i
done

echo "All processing complete"
