import argparse
from datetime import datetime
import os
from pathlib import Path
import furniture_bench  # noqa
from furniture_bench.sim_config import sim_config
import torch
import wandb
from ml_collections import ConfigDict
from src.eval.rollout import calculate_success_rate
from src.behavior.diffusion_policy import DiffusionPolicy
from src.dataset.normalizer import StateActionNormalizer
from src.gym import get_env

from ipdb import set_trace as bp  # noqa

api = wandb.Api()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--gpu", type=int, default=0)
    args.add_argument("--n-envs", type=int, default=1)
    args.add_argument("--n-rollouts", type=int, default=1)
    args.add_argument("--randomness", type=str, default="low")
    args.add_argument("--run-id", type=str, required=True)
    args.add_argument(
        "--furniture",
        "-f",
        type=str,
        choices=["one_leg", "lamp", "round_table"],
        required=True,
    )

    # Parse the arguments
    args = args.parse_args()
    # Get the model file and config
    run = api.run(f"robot-rearrangement/{args.run_id}")
    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(exist_ok=True).name

    config = ConfigDict(run.config)

    # Make the device
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Make the actor
    actor = DiffusionPolicy(
        device=device,
        encoder_name=config.vision_encoder.model,
        freeze_encoder=config.vision_encoder.freeze,
        normalizer=StateActionNormalizer(),
        config=config,
    )

    # Load the model weights
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    # Get the environment
    env = get_env(
        gpu_id=args.gpu,
        obs_type="image",
        furniture=args.furniture,
        num_envs=args.n_envs,
        randomness=args.randomness,
        resize_img=True,
        verbose=False,
    )

    # Prepare the rollout save directory
    rollout_save_dir = (
        Path(os.environ.get("DATA_DIR_RAW", "data"))
        / "raw"
        / "sim_rollouts"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    rollout_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving rollouts to {rollout_save_dir}")

    # Start a run to collect the results
    wandb.init(
        project="model-eval-test",
        entity="robot-rearrangement",
        job_type="eval",
        config=config.to_dict(),
        # mode="disabled",
    )

    # Perform the rollouts
    success_rate = calculate_success_rate(
        actor=actor,
        env=env,
        n_rollouts=args.n_rollouts,
        rollout_max_steps=sim_config["scripted_timeout"][args.furniture],
        epoch_idx=0,
        gamma=config.discount,
        rollout_save_dir=rollout_save_dir,
    )

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "n_rollouts": args.n_rollouts,
            "run_id": args.run_id,
            "n_success": round(success_rate * args.n_rollouts),
        }
    )

    # Close the run
    wandb.finish()
