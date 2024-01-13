import argparse
from datetime import datetime
import os
from pathlib import Path
import furniture_bench  # noqa
import torch
import wandb
from ml_collections import ConfigDict
from src.eval.rollout import calculate_success_rate
from src.behavior.idql import ImplicitQActor
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

    # Parse the arguments
    args = args.parse_args()
    # Get the model file and config
    run = api.run(f"robot-rearrangement/chef-training/{args.run_id}")
    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(exist_ok=True).name

    config = ConfigDict(run.config)

    # Then load in the critic module
    run_id = "1phr216o"
    critic_run = api.run(f"robot-rearrangement/critic-module-test/{run_id}")
    # config = ConfigDict(critic_run.config)
    config.update(critic_run.config)

    # Make the device
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Make the actor
    actor = ImplicitQActor(
        device=device,
        encoder_name=config.vision_encoder.model,
        freeze_encoder=config.vision_encoder.freeze,
        normalizer=StateActionNormalizer(),
        n_action_samples=5,
        config=config,
    )

    # Load the model weights
    actor.load_state_dict(torch.load(model_path), strict=False)
    actor.eval()


    # Load the critic weights
    model_wts = [file for file in critic_run.files() if file.name.endswith(".pt")][0]
    wts_path = model_wts.download(replace=False, exist_ok=True).name
    actor.critic_module.load_state_dict(torch.load(wts_path))

    config.policy = "idql"

    # Get the environment
    env = get_env(
        gpu_id=args.gpu,
        obs_type="image",
        furniture="one_leg",
        num_envs=args.n_envs,
        randomness=args.randomness,
        resize_img=True,
        verbose=False,
    )

    # Prepare the rollout save directory
    rollout_save_dir = (
        Path(os.environ.get("ROLLOUT_SAVE_DIR", "data"))
        / "raw"
        / "sim_rollouts"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    rollout_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving rollouts to {rollout_save_dir}")

    # Start a run to collect the results
    wandb.init(
        project="model-eval",
        entity="robot-rearrangement",
        group="idql",
        job_type="eval",
        config=config.to_dict(),
        # mode="disabled",
    )

    # Perform the rollouts
    success_rate = calculate_success_rate(
        actor=actor,
        env=env,
        n_rollouts=args.n_rollouts,
        rollout_max_steps=600,
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
