#!/bin/bash

mkdir -p out_logs


rand_level=med_perturb
load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
for val in {0..25};
do
    echo "python isaac_sim_raytrace.py --headless -i $val"
    python isaac_sim_raytrace.py --headless -i $val --sub-steps 6 --load-dir $load_dir --save --save-dir test_rerender/one_leg/$rand_level > out_logs/out_$val_$rand_level.log
done


rand_level=med
load_dir="/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/$rand_level/success"
for val in {0..25};
do
    echo "python isaac_sim_raytrace.py --headless -i $val"
    python isaac_sim_raytrace.py --headless -i $val --sub-steps 6 --load-dir $load_dir --save --save-dir test_rerender/one_leg/$rand_level > out_logs/out_$val_$rand_level.log
done
