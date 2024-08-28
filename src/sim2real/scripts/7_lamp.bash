#!/bin/bash

set -euo pipefail

save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/lamp_render_rppo_colors

# for val in {0..149};
# for val in {28..149};
for val in {92..149};
do
    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps 3 --load-dir /home/anthony/repos/research/robust-rearrangement/assembly-data/rerender/lp-rppo-dr-low-1/05nq024s/rerender/lp-rppo-dr-med-1/9068j0j9/raw/diffik/sim/lamp/rollout/med/success --furniture lamp --num-parts 3 --save --save-dir ${save_dir}/rollout/med/success -dr rand --part-random full --different-part-colors

    sleep 10

    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps 3 --load-dir /home/anthony/repos/research/robust-rearrangement/assembly-data/rerender/lp-rppo-dr-low-1/05nq024s/raw/diffik/sim/lamp/rollout/low/success --furniture lamp --num-parts 3 --save --save-dir ${save_dir}/rollout/low/success -dr rand --part-random full --different-part-colors

    sleep 10
done


save_dir=/home/anthony/repos/research/robust-rearrangement/assembly-data/raw/diffik/sim/lamp_render_demos_colors

for val in {0..24};
do
    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps 3 --load-dir /data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/lamp/teleop/med/success --furniture lamp --num-parts 3 --save --save-dir ${save_dir}/teleop/med/success -dr rand --part-random full --different-part-colors

    sleep 10

    python -m src.sim2real.isaac_sim_raytrace --headless -i $val --sub-steps 3 --load-dir /data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/lamp/teleop/med_perturb/success --furniture lamp --num-parts 3 --save --save-dir ${save_dir}/teleop/med_perturb/success -dr rand --part-random full --different-part-colors

    sleep 10
done
