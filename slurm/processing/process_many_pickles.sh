echo "Processing pickles for bc_unet_low, 250-500"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout \
    -r low -o success --n-cpus 16 --max-files 250 --offset 250 \
    --suffix bc_unet --output-suffix bc_low_250 --overwrite

echo "Processing pickles for bc_unet_low, 500-750"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout \
    -r low -o success --n-cpus 16 --max-files 250 --offset 500 \
    --suffix bc_unet --output-suffix bc_low_500 --overwrite

echo "Processing pickles for bc_unet_low, 750-1000"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout \
    -r low -o success --n-cpus 16 --max-files 250 --offset 750 \
    --suffix bc_unet --output-suffix bc_low_750 --overwrite

####

echo "Processing pickles for bc_unet_med, 500-750"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout \
    -r med -o success --n-cpus 16 --max-files 250 --offset 500 \
    --suffix bc_unet --output-suffix bc_med_500 --overwrite

echo "Processing pickles for bc_unet_med, 750-1000"
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s rollout \
    -r med -o success --n-cpus 16 --max-files 250 --offset 750 \
    --suffix bc_unet --output-suffix bc_med_750 --overwrite