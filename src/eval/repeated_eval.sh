#!/bin/bash

while true; do
    # python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/ofkmhwdm -f square_table --n-parts-assemble 2 --save-rollouts
    python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/09n7s3wo -f square_table --n-parts-assemble 2 --save-rollouts
    # python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 100 --randomness low --run-id multi-task/runs/otepvwjv -f square_table --n-parts-assemble 2 --save-rollouts
    # python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/fzwwjbe0 -f square_table --n-parts-assemble 2 --save-rollouts
    # python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/0o5ks4tu -f square_table --n-parts-assemble 2 --save-rollouts
done
