RUN_ID=ol-rppo-dr-low-1/kzlx4y3f
DATA_DIR_RAW=${DATA_DIR_RAW}/distill/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 512 \
    --n-rollouts 500000 \
    -f one_leg \
    --if-exists append \
    --max-rollout-steps 750 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness low \
    --save-rollouts --save-rollouts-suffix "" --break-on-n-success --stop-after-n-success 30000 

#     --save-rollouts --save-rollouts-suffix "" --break-on-n-success --stop-after-n-success 50000
