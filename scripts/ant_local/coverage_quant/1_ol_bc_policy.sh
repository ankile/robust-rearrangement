RUN_ID=ol-state-dr-1/r9wm1uo6
# RUN_ID=residual-ppo-dr-1/fj7ggmg7
DATA_DIR_RAW=${DATA_DIR_RAW}/coverage/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 1024 \
    --n-rollouts 1024 \
    -f one_leg \
    --if-exists append \
    --max-rollout-steps 650 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness med \
    --save-rollouts --save-failures
