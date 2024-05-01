# Loop 2 times.
for i in 1 2; do
    # Loop over each data subset.
    for data_subset in 10 20 30 40 50 100 200 300 500; do

        sbatch slurm/cannon/data_scaling/o_s_${data_subset}.sh
        sleep 5
    
    done
done