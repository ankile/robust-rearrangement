#!/bin/bash
set -euo pipefail

# declare -a offset_arr=(
# [0]=0
# [1]=1000
# [2]=2000
# [3]=3000
# [4]=4000
# [5]=5000
# [6]=6000
# [7]=7000
# [8]=8000
# [9]=9000
# )
delta=500
offset=0

for val in {0..19}
do
    # offset=${offset_arr[$val]}
    echo "val $val offset $offset"
    python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg_state_distill -s rollout -r low -o success --max-files $delta --offset $offset --n-cpus 4 --output-suffix $val-$delta
    offset=$(($offset + $delta))
done
