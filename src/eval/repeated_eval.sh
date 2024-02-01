#!/bin/bash

# while true; do
#     # Train end-to-end with SSRN encoder with image augmentation, 50 scripted demos only
#     python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 100 --randomness low --run-id baselines/runs/3edljp81 -f one_leg
    
#     # # Train end-to-end with SRN encoder without image augmentation, 50 scripted demos only
#     python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 100 --randomness low --run-id baselines/runs/udl4eaov -f one_leg

#     # Train with precomputed VIP features, 50 scripted demos only
#     python -m src.eval.evaluate_model --n-envs 20 --n-rollouts 100 --randomness low --run-id baselines/runs/oiwr1ods -f one_leg
# done


# Make a for loop over the different runs
for run_id in "3edljp81" "udl4eaov" "oiwr1ods"; do
    python -m src.eval.evaluate_model --n-envs 40 --n-rollouts 200 --randomness low --run-id baselines/runs/$run_id -f one_leg --wandb
done