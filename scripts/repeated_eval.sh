# Make an infinite that calls the same command over and over again:

while true;
do
    python -m src.eval.evaluate_model --run-id ol-state-dr-1/runs/klnpg80g --n-envs 32 --n-rollouts 32 -f one_leg --if-exists append --max-rollout-steps 650 --controller diffik --use-new-env --action-type pos --observation-space image --save-rollouts --randomness med
done
