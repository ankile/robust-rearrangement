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
from src.behavior import get_actor
from src.common.tasks import furniture2idx, task_timeout
from src.gym import get_env

from ipdb import set_trace as bp  # noqa
from wandb import Api
from wandb.sdk.wandb_run import Run

api = Api()


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
        choices=["one_leg", "lamp", "round_table", "desk", "square_table", "cabinet"],
        required=True,
    )
    args.add_argument("--save-rollouts", action="store_true")
    args.add_argument("--save-failures", action="store_true")
    args.add_argument("--wandb", action="store_true")
    args.add_argument("--n-parts-assemble", type=int, default=None)

    # Parse the arguments
    args = args.parse_args()
    # Get the model file and config
    run: Run = api.run(f"robot-rearrangement/{args.run_id}")
    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(exist_ok=True).name

    config = ConfigDict(run.config)

    if "act_rot_repr" not in config:
        config.act_rot_repr = "quat"

    # Add the original project name to the config
    config.project_name = run.project

    # Make the device
    device = torch.device(
        f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Make the actor
    actor = get_actor(config=config, device=device)

    # Load the model weights
    actor.load_state_dict(torch.load(model_path))
    actor.eval()

    # Set the timeout
    rollout_max_steps = task_timeout(args.furniture, n_parts=args.n_parts_assemble)

    # Get the environment
    env = get_env(
        gpu_id=args.gpu,
        furniture=args.furniture,
        num_envs=args.n_envs,
        randomness=args.randomness,
        max_env_steps=rollout_max_steps,
        resize_img=False,
        act_rot_repr=config.act_rot_repr,
        ctrl_mode="osc",
        verbose=False,
    )

    # Prepare the rollout save directory
    if args.save_rollouts:
        rollout_save_dir = (
            Path(os.environ["DATA_DIR_RAW"])
            / "raw"
            / "sim"
            / "rollout"
            / args.furniture
            / args.randomness
            / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        rollout_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving rollouts to {rollout_save_dir}")
    else:
        rollout_save_dir = None
        print("Not saving rollouts")

    # Start a run to collect the results
    wandb.init(
        project="model-eval-test",
        entity="robot-rearrangement",
        job_type="eval",
        config=config.to_dict(),
        mode="online" if args.wandb else "disabled",
        name=f"{run.name}-{run.id}",
    )

    # Perform the rollouts
    actor.set_task(furniture2idx[args.furniture])
    success_rate = calculate_success_rate(
        actor=actor,
        env=env,
        n_rollouts=args.n_rollouts,
        rollout_max_steps=rollout_max_steps,
        epoch_idx=0,
        gamma=config.discount,
        rollout_save_dir=rollout_save_dir,
        save_failures=args.save_failures,
        n_parts_assemble=args.n_parts_assemble,
    )

    print(
        f"Success rate: {success_rate:.2%} ({int(round(success_rate * args.n_rollouts))}/{args.n_rollouts})"
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

    # Remove the folder if we didn't save any rollouts
    if rollout_save_dir is not None and len(list(rollout_save_dir.iterdir())) == 0:
        rollout_save_dir.rmdir()
