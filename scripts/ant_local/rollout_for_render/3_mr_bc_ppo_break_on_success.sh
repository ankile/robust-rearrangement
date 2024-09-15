RUN_ID=mr-rppo-dr-low-1/dvw6zk8e
DATA_DIR_RAW=${DATA_DIR_RAW}/rerender/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 256 \
    --n-rollouts 10240 \
    -f mug_rack \
    --if-exists append \
    --max-rollout-steps 400 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness low \
    --save-rollouts --break-on-n-success --save-rollouts-suffix "" --stop-after-n-success 1000

