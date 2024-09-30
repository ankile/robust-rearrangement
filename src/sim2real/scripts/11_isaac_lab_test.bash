#!/bin/bash

set -euo pipefail

save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/render_rppo_test_isaac_lab
load_dir=/data/scratch-oc40/pulkitag/ankile/furniture-data/coverage/ol-state-dr-low-1/6i7hupje/raw/diffik/sim/one_leg/rollout/low/success 
val=1
python -m src.sim2real.isaac_lab_rerender -i $val --sub-steps 3 --load-dir $load_dir --furniture one_leg --save-dir ${save_dir}/rollout/low/success --num-parts 5
