echo "Processing pickles for rppo_1, files 0-250"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success --suffix rppo_1 --n-cpus 16 --max-files 250 --offset 0 --output-suffix rppo_1_000

echo "Processing pickles for rppo_1, files 250-500"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success --suffix rppo_1 --n-cpus 16 --max-files 250 --offset 250 --output-suffix rppo_1_250

echo "Processing pickles for bc_unet, files 0-250"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success --suffix bc_unet --n-cpus 16 --max-files 250 --offset 0 --output-suffix bc_unet_000

echo "Processing pickles for bc_unet, files 250-500"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout -r med -o success --suffix bc_unet --n-cpus 16 --max-files 250 --offset 250 --output-suffix bc_unet_250
