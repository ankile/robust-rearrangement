#!/bin/bash

# Base directories
LOAD_DIR="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/rollout/low"
SAVE_DIR="/data/scratch/ankile/robust-assembly-video-data/rendered/one_leg/rppo"

# Loop through indices
for i in $(seq 0 99); do
    echo "Processing index $i..."
    python -m src.sim2real.isaac_lab_rerender \
        --load-dir "$LOAD_DIR" \
        -i "$i" \
        --num-parts 5 \
        --furniture one_leg \
        -sub 3 \
        --save \
        --save-dir "$SAVE_DIR"
done

echo "Batch processing complete!"