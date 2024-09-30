RUN_ID=lp-rppo-dr-low-1/05nq024s
DATA_DIR_RAW=${DATA_DIR_RAW}/rerender/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 256 \
    --n-rollouts 10240 \
    -f lamp \
    --if-exists append \
    --max-rollout-steps 1000 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness low \
    --save-rollouts --break-on-n-success --save-rollouts-suffix "" --stop-after-n-success 150


RUN_ID=lp-rppo-dr-med-1/9068j0j9
DATA_DIR_RAW=${DATA_DIR_RAW}/rerender/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 256 \
    --n-rollouts 10240 \
    -f lamp \
    --if-exists append \
    --max-rollout-steps 1000 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness med \
    --save-rollouts --break-on-n-success --save-rollouts-suffix "" --stop-after-n-success 150

