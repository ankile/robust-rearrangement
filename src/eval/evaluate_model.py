import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import List
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument(
        "--furniture",
        "-f",
        type=str,
        choices=["one_leg", "lamp", "round_table", "desk", "square_table", "cabinet"],
        required=True,
    )
    parser.add_argument("--n-parts-assemble", type=int, default=None)

    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--save-failures", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--leaderboard", action="store_true")

    # Define what should be done if the success rate fields are already present
    parser.add_argument(
        "--if-exists",
        type=str,
        choices=["skip", "overwrite", "append", "error"],
        default="error",
    )
    parser.add_argument(
        "--run-state",
        type=str,
        default=None,
        choices=["running", "finished", "failed", "crashed"],
        nargs="*",
    )

    # For batch evaluating runs from a sweep
    parser.add_argument("--sweep-id", type=str, default=None)

    # Parse the arguments
    args = parser.parse_args()

    # Validate the arguments
    assert not (
        args.run_id and args.sweep_id
    ), "Only one of run_id or evaluate_sweep_id should be provided"
    assert args.run_state is None or all(
        [
            state in ["running", "finished", "failed", "crashed"]
            for state in args.run_state
        ]
    ), (
        "Invalid run-state: "
        f"{args.run_state}. Valid options are: None, running, finished, failed, crashed"
    )

    assert not args.leaderboard, "Leaderboard mode is not supported as of now"

    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Get the run(s) to test
    if args.sweep_id:
        runs: List[Run] = list(api.sweep(f"robot-rearrangement/{args.sweep_id}").runs)
    else:
        runs: List[Run] = [api.run(f"robot-rearrangement/{args.run_id}")]

    # Filter out the runs based on the run state
    if args.run_state:
        runs = [run for run in runs if run.state in args.run_state]

    print(f"Found {len(runs)} runs to evaluate")
    for run in runs:
        # Check if the run has already been evaluated
        how_update = None
        if run.summary.get("success_rate", None) is not None:
            if args.if_exists == "skip":
                print(f"Run: {run.name} has already been evaluated, skipping")
                continue
            elif args.if_exists == "error":
                raise ValueError(f"Run: {run.name} has already been evaluated")
            elif args.if_exists == "overwrite":
                print(f"Run: {run.name} has already been evaluated, overwriting")
                how_update = "overwrite"
            elif args.if_exists == "append":
                print(f"Run: {run.name} has already been evaluated, appending")
                how_update = "append"

        model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
        model_path = model_file.download(exist_ok=True).name

        # Get the current `test_epoch_loss` from the run
        test_epoch_loss = run.summary.get("test_epoch_loss", None)
        print(f"Evaluating run: {run.name} at test_epoch_loss: {test_epoch_loss}")

        # Create the config object with the project name and make it read-only
        config: DictConfig = OmegaConf.create(
            {**run.config, "project_name": run.project}, flags={"readonly": True}
        )

        # Get the normalizer
        normalizer_type = config.get("data", {}).get("normalization", "min_max")
        normalizer = get_normalizer(
            normalizer_type=normalizer_type, control_mode="delta"
        )

        # TODO: Fix this properly, but for now have an ugly escape hatch
        if isinstance(config.vision_encoder, str):
            # Read in the vision encoder config from the `vision_encoder` config group and set it
            OmegaConf.set_readonly(config, False)
            config.vision_encoder = OmegaConf.load(
                f"src/config/vision_encoder/{config.vision_encoder}.yaml"
            )
            OmegaConf.set_readonly(config, True)

            # Write it back to the run
            run.config = OmegaConf.to_container(config, resolve=True)
            run.update()

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

        # # Start a run to collect the results
        # if args.leaderboard:
        #     run = wandb.init(
        #         project="model-eval-test",
        #         entity="robot-rearrangement",
        #         job_type="eval",
        #         config=OmegaConf.to_container(config, resolve=True),
        #         mode="online" if args.wandb else "disabled",
        #         name=f"{run.name}-{run.id}",
        #     )
        # else:
        # run = wandb.init(
        #     project=run.project,
        #     entity="robot-rearrangement",
        #     config=OmegaConf.to_container(config, resolve=True),
        #     mode="online" if args.wandb else "disabled",
        #     id=run.id,
        #     resume="allow",
        # )

        save_dir = trajectory_save_dir(
            environment="sim",
            task=args.furniture,
            demo_source="rollout",
            randomness=args.randomness,
            create=False,
        )

        # Perform the rollouts
        actor.set_task(furniture2idx[args.furniture])
        rollout_stats = calculate_success_rate(
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
        success_rate = rollout_stats.success_rate

        print(
            f"Success rate: {success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})"
        )

        if args.wandb:
            print("Writing to wandb...")

            # Set the summary fields
            if how_update == "overwrite":
                run.summary["success_rate"] = success_rate
                run.summary["n_success"] = rollout_stats.n_success
                run.summary["n_rollouts"] = rollout_stats.n_rollouts
            elif how_update == "append":
                run.summary["n_success"] += rollout_stats.n_success
                run.summary["n_rollouts"] += rollout_stats.n_rollouts
                run.summary["success_rate"] = (
                    run.summary["n_success"] / run.summary["n_rollouts"]
                )
            else:
                raise ValueError(f"Invalid how_update: {how_update}")

            # Update the run to save the summary fields
            run.update()

        else:
            print("Not writing to wandb")
