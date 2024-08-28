# RUN_ID=lp-rppo-dr-med-1/ev23t35c
RUN_ID=lp-state-dr-med-1/fziwvs8k
DATA_DIR_RAW=${DATA_DIR_RAW}/gym_rerender_w_pert/${RUN_ID}

python -m src.eval.evaluate_model \
    --run-id ${RUN_ID} \
    --n-envs 32 \
    --n-rollouts 64 \
    -f lamp \
    --if-exists append \
    --max-rollout-steps 1000 \
    --controller diffik \
    --use-new-env \
    --action-type pos \
    --observation-space image \
    --randomness med \
    --save-rollouts --save-failures --save-rollouts-suffix ""


#     --save-rollouts --break-on-n-success --save-rollouts-suffix "" --stop-after-n-success 10
#     --n-rollouts 10240 \

#     --save-rollouts --save-failures --save-rollouts-suffix ""
#     --n-rollouts 64 \
