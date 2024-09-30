#!/bin/bash

set -euo pipefail

export PART_COLOR_BASE=white
save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/one_leg_render_rppo_colors
sub=3
furn=one_leg
num_parts=5
rand_level=med
run_name="residual-ppo-dr-med-1/h7dg0og4"
load_dir="/home/anthony/repos/research/robust-rearrangement/assembly-data/coverage/${run_name}/raw/diffik/sim/one_leg/rollout/med/rerender/success"

# for val in {0..170};
for val in {149..170};
do
    echo "Number $val, saving to $save_dir"
    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps $sub --load-dir $load_dir --furniture $furn --num-parts $num_parts --save --save-dir ${save_dir}/rollout/${rand_level}/success -dr rand --part-random full --different-part-colors

    sleep 10
done

export PART_COLOR_BASE=black
for val in {170..343};
do
    echo "Number $val, saving to $save_dir"
    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps $sub --load-dir $load_dir --furniture $furn --num-parts $num_parts --save --save-dir ${save_dir}/rollout/${rand_level}/success -dr rand --part-random full --different-part-colors

    sleep 10
done


export PART_COLOR_BASE=white
save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/one_leg_render_demos_colors
for val in {0..24};
do
    echo "Number $val, saving to $save_dir"
    rand_level=med
    load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps $sub --load-dir $load_dir --furniture $furn --num-parts $num_parts --save --save-dir ${save_dir}/teleop/${rand_level}/success -dr rand --part-random full --different-part-colors

    sleep 10

    rand_level=med_perturb
    load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps $sub --load-dir $load_dir --furniture $furn --num-parts $num_parts --save --save-dir ${save_dir}/teleop/${rand_level}/success -dr rand --part-random full --different-part-colors

    sleep 10
done
