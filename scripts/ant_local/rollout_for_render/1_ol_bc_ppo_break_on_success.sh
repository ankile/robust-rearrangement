RUN_ID=residual-ppo-dr-med-1/h7dg0og4
DATA_DIR_RAW=${DATA_DIR_RAW}/coverage/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 256 \
    --n-rollouts 10240 \
    -f one_leg \
    --if-exists append \
    --max-rollout-steps 750 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness med --visualize # \
    # --save-rollouts --break-on-n-success --save-rollouts-suffix rerender --stop-after-n-success 250

