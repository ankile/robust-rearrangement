#!/bin/bash

while true; do
    # Train end-to-end with SRN encoder with image augmentation, 50 teleop demos only
    python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/09n7s3wo -f square_table --n-parts-assemble 2 --save-rollouts
    
    # Train end-to-end with SRN encoder with image augmentation, 67 teleop demos only
    python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/0o5ks4tu -f square_table --n-parts-assemble 2 --save-rollouts

    # Train with precomputed VIP features on the data we have across tasks
    python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/cpxsvq4m -f square_table --n-parts-assemble 2 --save-rollouts
    
    # Train with precomputed VIP features on only teleop data, 66 demos
    python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 20 --randomness low --run-id teleop-finetune/runs/dy8shhlw -f square_table --n-parts-assemble 2 --save-rollouts
done
