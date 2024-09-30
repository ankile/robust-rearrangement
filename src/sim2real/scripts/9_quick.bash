#!/bin/bash

set -euo pipefail

save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/one_leg_bc_model_talk_render
#load_dir=/data/scratch-oc40/pulkitag/ankile/furniture-data/coverage/ol-state-dr-low-1/6i7hupje/raw/diffik/sim/one_leg/rollout/low/success 
load_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/coverage/ol-state-dr-med-1/9zjnzg4r/raw/diffik/sim/one_leg/rollout/med/failure
val=1
python -m src.sim2real.isaac_sim_raytrace -i $val --sub-steps 3 --load-dir $load_dir --furniture one_leg --num-parts 5 # --save --save-dir ${save_dir}/rollout/low/failure 


# --headless 
