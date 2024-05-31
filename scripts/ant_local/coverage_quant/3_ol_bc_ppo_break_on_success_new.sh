# export WANDB_ENTITY=robust-rearrangement

# RUN_ID=ol-state-dr-med-1/9zjnzg4r
# RUN_ID=residual-ppo-dr-1/7mv6o4i9
# RUN_ID=residual-ppo-dr-low-1/8z64fcnf

RUN_ID=residual-ppo-dr-med-1/h7dg0og4
# RUN_ID=residual-ppo-dr-med-1/u1icj0g9
DATA_DIR_RAW=${DATA_DIR_RAW}/coverage/${RUN_ID}

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
    --randomness med \
    --save-rollouts --save-failures --save-rollouts-suffix large_success --break-on-n-success --stop-after-n-success 10000 --record-for-coverage

#     --wt-type eval_best \

