#!/bin/bash

set -euo pipefail

save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/render_rppo_test_isaac_lab
load_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/gym_rerender_no_pert_finger_pos_mode/ol-rppo-dr-med-1/2610ys6o/raw/diffik/sim/one_leg/rollout/med/success

for val in {1..10}
# val=1
do
    python -m src.sim2real.isaac_lab_rerender -i $val --sub-steps 3 --load-dir $load_dir --furniture one_leg --save --save-dir ${save_dir}/rollout/low/success --num-parts 5 
done
