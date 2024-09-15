#!/bin/bash

mkdir -p out_logs

sub=2

# for copy in {2..5};
# do
#     for val in {0..25};
#     do
#         # domain rand - base/base/same
#         part_rand=base
#         table_rand=base
# 
#         rand_level=med_perturb
#         load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
#         echo "Demo $val load dir $load_dir base/base"
#         python isaac_sim_raytrace.py \
#             --headless -i $val \
#             --sub-steps $sub \
#             --load-dir $load_dir \
#             --save \
#             --save-dir test_rerender/one_leg/domain_rand/${rand_level}_${part_rand}_${table_rand}_copy_${copy} \
#             -dr rand --part-random $part_rand --table-random $table_rand > out_logs/out_${val}_${rand_level}_${part_rand}_${table_rand}.log
# 
#         rand_level=med
#         load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
#         echo "Demo $val load dir $load_dir base/base"
#         python isaac_sim_raytrace.py \
#             --headless -i $val \
#             --sub-steps $sub \
#             --load-dir $load_dir \
#             --save \
#             --save-dir test_rerender/one_leg/domain_rand/${rand_level}_${part_rand}_${table_rand}_copy_${copy} \
#             -dr rand --part-random $part_rand --table-random $table_rand > out_logs/out_${val}_${rand_level}_${part_rand}_${table_rand}.log
#     done
# done


for copy in {4..5};
do
    for val in {0..25};
    do
        # domain rand - full/full/same
        part_rand=full
        table_rand=full

        rand_level=med_perturb
        load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
        echo "Demo $val load dir $load_dir full/full"
        python isaac_sim_raytrace.py \
            --headless -i $val \
            --sub-steps $sub \
            --load-dir $load_dir \
            --save \
            --save-dir test_rerender/one_leg/domain_rand/${rand_level}_${part_rand}_${table_rand}_copy_${copy} \
            -dr rand --part-random $part_rand --table-random $table_rand > out_logs/out_${val}_${rand_level}_${part_rand}_${table_rand}.log

        rand_level=med
        load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
        echo "Demo $val load dir $load_dir full/full"
        python isaac_sim_raytrace.py \
            --headless -i $val \
            --sub-steps $sub \
            --load-dir $load_dir \
            --save \
            --save-dir test_rerender/one_leg/domain_rand/${rand_level}_${part_rand}_${table_rand}_copy_${copy} \
            -dr rand --part-random $part_rand --table-random $table_rand > out_logs/out_${val}_${rand_level}_${part_rand}_${table_rand}.log

    done
done


# for copy in {0..5};
# do
#     for val in {0..25};
#     do
#         # domain rand - full/full/diff
#         part_rand=full
#         table_rand=full
# 
#         rand_level=med_perturb
#         load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
#         echo "Demo $val load dir $load_dir full/full/diff"
#         python isaac_sim_raytrace.py \
#             --headless -i $val \
#             --sub-steps $sub \
#             --load-dir $load_dir \
#             --save \
#             --save-dir test_rerender/one_leg/domain_rand/${rand_level}_${part_rand}_${table_rand}_diffparts_copy_${copy} \
#             -dr rand --part-random $part_rand --table-random $table_rand --different-part-colors > out_logs/out_${val}_${rand_level}_${part_rand}_${table_rand}.log
# 
#         rand_level=med
#         load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
#         echo "Demo $val load dir $load_dir full/full/diff"
#         python isaac_sim_raytrace.py \
#             --headless -i $val \
#             --sub-steps $sub \
#             --load-dir $load_dir \
#             --save \
#             --save-dir test_rerender/one_leg/domain_rand/${rand_level}_${part_rand}_${table_rand}_diffparts_copy_${copy} \
#             -dr rand --part-random $part_rand --table-random $table_rand --different-part-colors > out_logs/out_${val}_${rand_level}_${part_rand}_${table_rand}.log
#     done
# done
