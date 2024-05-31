# RUN_ID=ol-state-dr-med-1/9zjnzg4r
# RUN_ID=residual-ppo-dr-1/7mv6o4i9
RUN_ID=residual-ppo-dr-low-1/8z64fcnf
DATA_DIR_RAW=${DATA_DIR_RAW}/coverage/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 256 \
    --n-rollouts 10240 \
    -f one_leg \
    --if-exists append \
    --max-rollout-steps 700 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space state \
    --randomness low \
    --save-rollouts --save-failures --break-on-n-success --stop-after-n-success 1024

