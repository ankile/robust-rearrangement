# Make an infinite loop in this .sh file that runs the sync script
while true; do
    sleep 900
    python -m src.data_processing.sync -d -s processed/diffik/sim/one_leg -y
done
