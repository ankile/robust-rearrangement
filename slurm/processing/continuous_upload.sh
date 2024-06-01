
# Make an infinite loop in this .sh file that runs the sync script
while true; do
    python -m src.data_processing.sync -u -s processed/diffik/sim/one_leg -D -y
    sleep 1800
done