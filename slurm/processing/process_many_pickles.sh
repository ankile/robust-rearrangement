echo "Processing pickles for rppo_2, files 0-250"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success \
    --suffix rppo_2 --n-cpus 16 --max-files 250 --offset 0 --output-suffix rppo_2_000

echo "Processing pickles for rppo_2, files 250-500"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success \
    --suffix rppo_2 --n-cpus 16 --max-files 250 --offset 250 --output-suffix rppo_2_250

echo "Processing pickles for rppo_2, files 500-750"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success \
    --suffix rppo_2 --n-cpus 16 --max-files 250 --offset 500 --output-suffix rppo_2_500

echo "Processing pickles for rppo_2, files 750-1000"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success \
    --suffix rppo_2 --n-cpus 16 --max-files 250 --offset 750 --output-suffix rppo_2_750