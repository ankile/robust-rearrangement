offset=91000
increment=1000
iterations=9

for ((i=1; i<=iterations; i++))
do
    python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg_state_distill_large -s rollout -r low -o success --n-cpus 8 --max-files 1000 --offset $offset --output-suffix rl_state_$offset
    offset=$((offset + increment))
done
