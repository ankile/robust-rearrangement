import argparse
from datetime import datetime
import os
from pathlib import Path
import furniture_bench  # noqa
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import furniture2idx, task_timeout
from src.common.files import trajectory_save_dir
from src.gym import get_env
from src.dataset import get_normalizer

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
    args.add_argument("--leaderboard", action="store_true")

    # Parse the arguments
    args = args.parse_args()
    # Get the model file and config
    run: Run = api.run(f"robot-rearrangement/{args.run_id}")
    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(exist_ok=True).name

    # Get the current `test_epoch_loss` from the run
    test_epoch_loss = run.summary.get("test_epoch_loss", None)
    print(f"Evaluating run: {run.name} at test_epoch_loss: {test_epoch_loss}")

    # Create the config object with the project name and make it read-only
    config: DictConfig = OmegaConf.create(
        {**run.config, "project_name": run.project}, flags={"readonly": True}
    )

    # Make the device
    device = torch.device(
        f"cuda:{config.training.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    # Get the normalizer
    normalizer_type = config.get("data", {}).get("normalization", "min_max")
    normalizer = get_normalizer(normalizer_type=normalizer_type, control_mode="delta")

    # Make the actor
    actor = get_actor(config=config, normalizer=normalizer, device=device)

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
        act_rot_repr=config.control.act_rot_repr,
        ctrl_mode="osc",
        action_type="delta",
        verbose=False,
    )

    # Start a run to collect the results
    if args.leaderboard:
        run = wandb.init(
            project="model-eval-test",
            entity="robot-rearrangement",
            job_type="eval",
            config=OmegaConf.to_container(config, resolve=True),
            mode="online" if args.wandb else "disabled",
            name=f"{run.name}-{run.id}",
        )
    else:
        run = wandb.init(
            project=run.project,
            entity="robot-rearrangement",
            config=OmegaConf.to_container(config, resolve=True),
            mode="online" if args.wandb else "disabled",
            id=run.id,
            resume="allow",
        )

    save_dir = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source="rollout",
        randomness=args.randomness,
        create=False,
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
        rollout_save_dir=save_dir,
        save_failures=args.save_failures,
        n_parts_assemble=args.n_parts_assemble,
    )

    print(
        f"Success rate: {success_rate:.2%} ({int(round(success_rate * args.n_rollouts))}/{args.n_rollouts})"
    )

    # Log the success rate to wandb
    run.log(
        {
            "success_rate": success_rate,
            "n_rollouts": args.n_rollouts,
            "run_id": args.run_id,
            "n_success": round(success_rate * args.n_rollouts),
            "test_epoch_loss_at_eval": test_epoch_loss,
        }
    )
