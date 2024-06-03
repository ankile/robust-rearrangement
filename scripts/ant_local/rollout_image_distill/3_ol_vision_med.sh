RUN_ID=ol-vision-scaling-med-1/dsr3o3pf
DATA_DIR_RAW=${DATA_DIR_RAW}/distill/${RUN_ID}

while true; 
do
    python -m src.eval.evaluate_model \
        --run-id ${RUN_ID} \
        --n-envs 32 \
        --n-rollouts 1000 \
        -f one_leg \
        --if-exists append \
        --max-rollout-steps 750 \
        --controller diffik \
        --use-new-env \
        --action-type pos \
        --observation-space image \
        --randomness med \
        --wt-type best_success_rate \
        --save-rollouts --save-rollouts-suffix "" --break-on-n-success --stop-after-n-success 100
    sleep 10
done
