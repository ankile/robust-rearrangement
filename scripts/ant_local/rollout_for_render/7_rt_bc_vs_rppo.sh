RUN_ID=rt-rppo-dr-med-1/k737s8lj
DATA_DIR_RAW=${DATA_DIR_RAW}/gym_rerender_no_pert/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 32 \
    --n-rollouts 10240 \
    -f round_table \
    --if-exists append \
    --max-rollout-steps 1000 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space image \
    --randomness med \
    --save-rollouts --break-on-n-success --save-rollouts-suffix "" --stop-after-n-success 100

