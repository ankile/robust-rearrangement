#!/bin/bash

sub=3
part_rand=base
table_rand=base
rand_level=med
run_name="residual-ppo-dr-med-1/h7dg0og4"
load_dir="/home/anthony/repos/research/robust-rearrangement/assembly-data/coverage/${run_name}/raw/diffik/sim/one_leg/rollout/med/rerender/success"

for val in {0..343};
do
    echo "Demo $val load dir $load_dir part rand level $part_rand table rand level $table_rand"
    python -m src.sim2real.isaac_sim_raytrace \
        --headless -i $val \
        --sub-steps $sub \
        --load-dir $load_dir \
        --save \
        --save-dir rerender/one_leg/domain_rand/${run_name}/rollout/${rand_level}_${part_rand}_${table_rand} \
        -dr rand --part-random $part_rand --table-random $table_rand # > out_logs/out_${val}_${rand_level}_${part_rand}_${table_rand}.log
done
