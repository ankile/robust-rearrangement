# Loop 2 times.
for i in 1 2; do
    # Loop over each data subset.
    for data_subset in 10 20 30 40 50; do

        sbatch slurm/supercloud/one_leg/data_scaling/o_s_${data_subset}.sh
        sleep 120
    
    done
done