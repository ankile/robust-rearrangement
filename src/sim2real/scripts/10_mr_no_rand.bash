#!/bin/bash

set -euo pipefail

save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/mug_rack_render_rppo_fixed_part_colors
sub=3
furn=mug_rack
num_parts=2
rand_level=low
run_name="mr-rppo-dr-low-1/dvw6zk8e"
load_dir="/home/anthony/repos/research/robust-rearrangement/assembly-data/rerender/mr-rppo-dr-low-1/dvw6zk8e/raw/diffik/sim/mug_rack/rollout/low/success"

for val in {0..1000};
do
    echo "Number $val, saving to $save_dir"
    python -m src.sim2real.isaac_sim_raytrace --headless -i $val \
        --sub-steps $sub \
        --load-dir $load_dir \
        --furniture $furn \
        --num-parts $num_parts # \

        # --save --save-dir ${save_dir}/rollout/${rand_level}/success \
        # -dr rand 

    sleep 10
done
