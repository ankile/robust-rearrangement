#!/bin/bash

export PART_COLOR_BASE=black
sub=3
part_rand=base
table_rand=base

rand_level=med
run_name="residual-ppo-dr-med-1/h7dg0og4"
load_dir="/home/anthony/repos/research/robust-rearrangement/assembly-data/coverage/${run_name}/raw/diffik/sim/one_leg/rollout/med/rerender/success"

# for val in {0..25};
# for val in {40..343};
for val in {113..343};
do
    echo "Demo $val load dir $load_dir part rand level $part_rand table rand level $table_rand"
    # rand_level=med
    # load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"

    python -m src.sim2real.isaac_sim_raytrace \
        --headless -i $val \
        --sub-steps $sub \
        --load-dir $load_dir \
        --save \
        --save-dir rerender/one_leg/domain_rand/${run_name}/rollout/${rand_level}_${part_rand}_${table_rand}_bright_black \
        -dr rand --part-random $part_rand --table-random $table_rand 

    sleep 60

    # rand_level=med_perturb
    # load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
    # python -m src.sim2real.isaac_sim_raytrace \
    #     --headless -i $val \
    #     --sub-steps $sub \
    #     --load-dir $load_dir \
    #     --save \
    #     --save-dir rerender/one_leg/domain_rand/demos/${rand_level}_${part_rand}_${table_rand}_bright_black \
    #     -dr rand --part-random $part_rand --table-random $table_rand

#         --headless -i $val \
#         -i $val \
#         --save-dir rerender/one_leg/domain_rand/demos/${rand_level}_${part_rand}_${table_rand}_bright_black \
#         --save-dir rerender/one_leg/domain_rand/${run_name}/rollout/${rand_level}_${part_rand}_${table_rand} \
done
