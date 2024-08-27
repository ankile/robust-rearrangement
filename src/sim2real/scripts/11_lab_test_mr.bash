#!/bin/bash

set -euo pipefail

# save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/mug_rack_render_rppo_test_isaac_lab
# sub=3
# furn=mug_rack
# num_parts=2
# rand_level=low
# run_name="mr-rppo-dr-low-1/dvw6zk8e"
# load_dir="/home/anthony/repos/research/robust-rearrangement/assembly-data/rerender/mr-rppo-dr-low-1/dvw6zk8e/raw/diffik/sim/mug_rack/rollout/low/success"
# 
# for val in {0..0};
# do
#     echo "Number $val, saving to $save_dir"
#     python -m src.sim2real.isaac_lab_rerender -i $val \
#         --sub-steps $sub \
#         --load-dir $load_dir \
#         --furniture $furn \
#         --enable_cameras \
#         --save-dir ${save_dir}/rollout/${rand_level}/success \
#         --num-parts $num_parts # \ 
#         # -dr # \
#         # --save --save-dir ${save_dir}/rollout/${rand_level}/success 
# 
#     sleep 10
# done

save_dir=./render_rppo_test_isaac_lab
load_dir=/data/scratch-oc40/pulkitag/ankile/furniture-data/coverage/ol-state-dr-low-1/6i7hupje/raw/diffik/sim/one_leg/rollout/low/success 
val=1
python -m src.sim2real.isaac_lab_rerender -i $val --sub-steps 3 --load-dir $load_dir --furniture one_leg --save-dir ${save_dir}/rollout/low/success --num-parts 5 # --save --save-dir ${save_dir}/rollout/low/failure 


        # -dr # rand --part-random base --table-random base

        # --domain-rand \

    # python -m src.sim2real.isaac_lab_rerender -i $val \
    # python -m src.sim2real.isaac_lab_rerender --headless -i $val \
