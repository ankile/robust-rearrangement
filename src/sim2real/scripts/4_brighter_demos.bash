#!/bin/bash

sub=3
part_rand=base
table_rand=base

# for val in {0..25};
for val in {13..25};
do
    echo "Demo $val load dir $load_dir part rand level $part_rand table rand level $table_rand"
    rand_level=med
    load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
    python -m src.sim2real.isaac_sim_raytrace \
        --headless -i $val \
        --sub-steps $sub \
        --load-dir $load_dir \
        --save \
        --save-dir rerender/one_leg/domain_rand/demos/${rand_level}_${part_rand}_${table_rand}_bright \
        -dr rand --part-random $part_rand --table-random $table_rand 

    rand_level=med_perturb
    load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
    python -m src.sim2real.isaac_sim_raytrace \
        --headless -i $val \
        --sub-steps $sub \
        --load-dir $load_dir \
        --save \
        --save-dir rerender/one_leg/domain_rand/demos/${rand_level}_${part_rand}_${table_rand}_bright \
        -dr rand --part-random $part_rand --table-random $table_rand

#         --headless -i $val \
#         -i $val \
done
