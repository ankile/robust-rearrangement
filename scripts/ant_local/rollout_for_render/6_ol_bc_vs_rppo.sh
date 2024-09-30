RUN_ID=ol-rppo-dr-med-1/2610ys6o
DATA_DIR_RAW=${DATA_DIR_RAW}/gym_rerender_no_pert_finger_pos_mode/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 32 \
    --n-rollouts 10240 \
    -f one_leg \
    --if-exists append \
    --max-rollout-steps 750 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness med \
    --save-rollouts --break-on-n-success --save-rollouts-suffix "" --stop-after-n-success 10

