import argparse
import os
from pathlib import Path
import time

import furniture_bench
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
import torch  # needs to be after isaac gym imports
from omegaconf import DictConfig, OmegaConf
from src.behavior.base import Actor  # noqa
from src.behavior.diffusion import DiffusionPolicy  # noqa
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import task2idx, task_timeout
from src.common.files import trajectory_save_dir
from src.gym import get_rl_env
from src.eval.eval_utils import load_model_weights

from typing import Any, List, Optional
from ipdb import set_trace as bp  # noqa
import wandb
from wandb import Api
from wandb.sdk.wandb_run import Run

api = Api(overrides=dict(entity=os.environ.get("WANDB_ENTITY")))


class LocalCheckpointWrapper:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)

        self.checkpoint = torch.load(self.checkpoint_path)

        self.config: DictConfig = OmegaConf.create(self.checkpoint["config"])
        self.name = self.checkpoint_path.stem
        self.id = self.name
        self.project = "local_evaluation"
        self.entity = "local"
        self.summary = {}

    @property
    def state(self):
        return "finished"

    def update(self):
        # In a real WandB run, this would push updates to the server
        # For local evaluation, we'll just save the summary to a JSON file
        import json

        summary_path = self.checkpoint_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f"Updated summary saved to {summary_path}")

    def file(self, name: str):
        # This method would normally return a WandB file object
        # For local evaluation, we'll return a dummy object with a download method
        class DummyFile:
            def __init__(self, path):
                self.path = path
                self.name = os.path.basename(path)

            def download(self, replace=True):
                # For local files, we don't need to download anything
                pass

        return DummyFile(self.checkpoint_path)

    def files(self):
        # This method would normally return an iterator of WandB file objects
        # For local evaluation, we'll return an iterator with just the checkpoint file
        yield self.file(self.checkpoint_path.name)

    def get(self, key: str, default: Any = None) -> Any:
        # This method mimics the behavior of wandb.run.config.get()
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        # This method allows accessing config items using square bracket notation
        return self.config[key]


def validate_args(args: argparse.Namespace):
    assert (
        sum(
            [
                args.run_id is not None,
                args.sweep_id is not None,
                args.project_id is not None,
                args.wt_path is not None,
            ]
        )
        == 1
    ), "Exactly one of run-id, sweep-id, project-id must be provided"
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

    assert not args.store_video_wandb or args.wandb, "store-video-wandb requires wandb"


def get_runs(args: argparse.Namespace) -> List[Run]:
    # Clear the cache to make sure we get the latest runs
    if args.wt_path:

        run = LocalCheckpointWrapper(args.wt_path)
        runs = [run]

    else:

        api.flush()
        if args.sweep_id:
            runs: List[Run] = list(api.sweep(args.sweep_id).runs)
        elif args.run_id:
            runs: List[Run] = [api.run(run_id) for run_id in args.run_id]
        elif args.project_id:
            runs: List[Run] = list(api.runs(args.project_id))
        else:
            raise ValueError

        # Filter out the runs based on the run state
        if args.run_state:
            runs = [run for run in runs if run.state in args.run_state]

    return runs


def vision_encoder_field_hotfix(run, config):
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


def convert_state_dict(state_dict):
    if not any(k.startswith("encoder1.0") for k in state_dict.keys()) and not any(
        k.startswith("encoder1.model.nets.3") for k in state_dict.keys()
    ):
        print("Dict already in the correct format")
        return

    # Change all instances of "encoder1.0" to "encoder1" and "encoder2.0" to "encoder2"
    # and all instances of "encoder1.1" to encoder1_proj and "encoder2.1" to "encoder2_proj"
    for k in list(state_dict.keys()):
        if k.startswith("encoder1.0"):
            new_k = k.replace("encoder1.0", "encoder1")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.0"):
            new_k = k.replace("encoder2.0", "encoder2")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder1.1"):
            new_k = k.replace("encoder1.1", "encoder1_proj")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.1"):
            new_k = k.replace("encoder2.1", "encoder2_proj")
            state_dict[new_k] = state_dict.pop(k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=False, nargs="*")
    parser.add_argument("--wt-path", type=str, default=None)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument(
        "--task",
        "-f",
        type=str,
        choices=[
            "one_leg",
            "lamp",
            "round_table",
            "desk",
            "square_table",
            "cabinet",
            "mug_rack",
            "factory_peg_hole",
        ],
        required=True,
    )
    parser.add_argument("--n-parts-assemble", type=int, default=None)

    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--store-full-resolution-video", action="store_true")

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

    # For batch evaluating runs from a sweep or a project
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)

    parser.add_argument("--continuous-mode", action="store_true")
    parser.add_argument(
        "--continuous-interval",
        type=int,
        default=60,
        help="Pause interval before next evaluation",
    )
    parser.add_argument("--ignore-currently-evaluating-flag", action="store_true")

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--store-video-wandb", action="store_true")
    parser.add_argument("--eval-top-k", type=int, default=None)
    parser.add_argument(
        "--action-type", type=str, default="pos", choices=["delta", "pos", "relative"]
    )
    parser.add_argument("--prioritize-fewest-rollouts", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--compress-pickles", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--april-tags", action="store_true")

    parser.add_argument(
        "--observation-space", choices=["image", "state"], default="state"
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--wt-type", type=str, default="best_success_rate")

    parser.add_argument("--stop-after-n-success", type=int, default=0)
    parser.add_argument("--break-on-n-success", action="store_true")
    parser.add_argument("--record-for-coverage", action="store_true")

    parser.add_argument("--save-rollouts-suffix", type=str, default="")

    # Parse the arguments
    args = parser.parse_args()

    # Validate the arguments
    validate_args(args)

    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Set the timeout
    rollout_max_steps = (
        task_timeout(args.task, n_parts=args.n_parts_assemble)
        if args.max_rollout_steps is None
        else args.max_rollout_steps
    )

    # Get the environment
    # TODO: This needs to be changed to enable recreation the env for each run
    print(
        f"Creating the environment with action_type {args.action_type} (this needs to be changed to enable recreation the env for each run)"
    )
    env: Optional[FurnitureSimEnv] = None

    f: str = args.task

    # Summary prefix, shoprtened to spf for brevity downstream
    spf = f"{f}/" + "" if args.multitask else ""

    # Start the evaluation loop
    print(f"Starting evaluation loop in continuous mode: {args.continuous_mode}")
    try:
        while True:
            # Get the run(s) to test
            runs = get_runs(args)

            # For now, filter out only the runs with strictly positive success rates to add more runs to them to get a better estimate
            if args.eval_top_k is not None:
                # Get the top k runs
                runs = sorted(
                    runs,
                    key=lambda run: run.summary.get(spf + "success_rate", 0),
                    reverse=True,
                )[: args.eval_top_k]

            # Also, evaluate the ones with the fewest rollouts first (if they have any)
            runs = sorted(
                runs,
                key=lambda run: run.summary.get(spf + "n_rollouts", 0),
            )

            print(f"Found {len(runs)} runs to evaluate:")
            for run in runs:
                print(
                    f"    Run: {run.name}: {run.summary.get(spf + 'n_rollouts', 0)}, {run.summary.get(spf + 'success_rate', None)}"
                )
            for run in runs:
                # First, we must flush the api and request the run again in case the information is stale
                if not args.wt_path:
                    api.flush()
                    run = api.run("/".join([run.project, run.id]))

                # Check if the run is currently being evaluated
                if (
                    run.config.get("currently_evaluating", False)
                    and not args.ignore_currently_evaluating_flag
                ):
                    print(f"Run: {run.name} is currently being evaluated, skipping")
                    continue

                # Check if the number of rollouts this run has is greater than the max_rollouts
                if args.max_rollouts is not None:
                    if run.summary.get(spf + "n_rollouts", 0) >= args.max_rollouts:
                        print(
                            f"Run: {run.name} has already been evaluated {run.summary.get(spf + 'n_rollouts', 0)} times, skipping"
                        )
                        continue

                # Check if the run has already been evaluated
                how_update = "overwrite"
                if run.summary.get(spf + "success_rate", None) is not None:
                    if args.if_exists == "skip":
                        print(f"Run: {run.name} has already been evaluated, skipping")
                        continue
                    elif args.if_exists == "error":
                        raise ValueError(f"Run: {run.name} has already been evaluated")
                    elif args.if_exists == "overwrite":
                        print(
                            f"Run: {run.name} has already been evaluated, overwriting"
                        )
                        how_update = "overwrite"
                    elif args.if_exists == "append":
                        print(f"Run: {run.name} has already been evaluated, appending")
                        how_update = "append"

                # If in overwrite set the currently_evaluating flag to true runs can cooperate better in skip mode
                if args.wandb:
                    print(
                        f"Setting currently_evaluating flag to true for run: {run.name}"
                    )
                    run.config["currently_evaluating"] = True
                    run.update()

                # Get the current `test_epoch_loss` from the run
                test_epoch_loss = run.summary.get("test_epoch_loss", None)
                print(
                    f"Evaluating run: {run.name} at test_epoch_loss: {test_epoch_loss}"
                )

                cfg = OmegaConf.create(run.config)

                # Check that we didn't set the wrong action type and pose representation
                assert cfg.control.control_mode == args.action_type

                print(OmegaConf.to_yaml(cfg))

                # Make the actor
                actor: Actor = get_actor(cfg=cfg, device=device)

                # Set the inference steps of the actor
                if isinstance(actor, DiffusionPolicy):
                    actor.inference_steps = 4

                if args.wt_path:
                    actor.load_state_dict(run.checkpoint["model_state_dict"])
                    actor.eval()
                    actor.to(device)

                else:
                    actor: Optional[Actor] = load_model_weights(
                        run=run, actor=actor, wt_type=args.wt_type
                    )

                if actor is None:
                    print(
                        f"Skipping run: {run.name} as no weights for wt_type: {args.wt_type} was not found"
                    )
                    continue

                save_dir = (
                    trajectory_save_dir(
                        controller="diffik",
                        domain="sim",
                        task=args.task,
                        demo_source="rollout",
                        randomness=args.randomness,
                        suffix=args.save_rollouts_suffix,
                        create=True,
                    )
                    if args.save_rollouts
                    else None
                )

                if args.store_video_wandb:
                    # For the run table with videos to be saved to wandb,
                    # a run needs to be active, so we initialie run here
                    wandb.init(
                        project=run.project,
                        entity=run.entity,
                        id=run.id,
                        resume="allow",
                    )

                # Only actually load the environment after we know we've got at least one run to evaluate
                if env is None:
                    env = get_rl_env(
                        gpu_id=args.gpu,
                        task=args.task,
                        num_envs=args.n_envs,
                        randomness=args.randomness,
                        observation_space=args.observation_space,
                        max_env_steps=5_000,
                        resize_img=False,
                        act_rot_repr="rot_6d",
                        action_type=args.action_type,
                        april_tags=args.april_tags,
                        verbose=args.verbose,
                        headless=not args.visualize,
                    )

                # Perform the rollouts
                print(f"Starting rollout of run: {run.name}")
                actor.set_task(task2idx[args.task])
                rollout_stats = calculate_success_rate(
                    actor=actor,
                    env=env,
                    n_rollouts=args.n_rollouts,
                    rollout_max_steps=rollout_max_steps,
                    epoch_idx=0,
                    discount=cfg.discount,
                    rollout_save_dir=save_dir,
                    save_failures=args.save_failures,
                    n_parts_assemble=args.n_parts_assemble,
                    compress_pickles=args.compress_pickles,
                    resize_video=not args.store_full_resolution_video,
                    break_on_n_success=args.break_on_n_success,
                    stop_after_n_success=args.stop_after_n_success,
                    record_first_state_only=args.record_for_coverage,
                )

                if args.store_video_wandb:
                    # Close the run to save the videos
                    wandb.finish()

                success_rate = rollout_stats.success_rate

                print(
                    f"Success rate: {success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})"
                )

                if args.wandb:
                    print("Writing to wandb...")

                    s: dict = run.summary

                    # Set the summary fields
                    if how_update == "overwrite":
                        s[spf + "success_rate"] = success_rate
                        s[spf + "n_success"] = rollout_stats.n_success
                        s[spf + "n_rollouts"] = rollout_stats.n_rollouts
                        s[spf + "total_return"] = rollout_stats.total_return
                        s[spf + "average_return"] = (
                            rollout_stats.total_return / rollout_stats.n_rollouts
                        )
                        s[spf + "total_reward"] = rollout_stats.total_reward
                        s[spf + "average_reward"] = (
                            rollout_stats.total_reward / rollout_stats.n_rollouts
                        )
                    elif how_update == "append":
                        s[spf + "n_success"] += rollout_stats.n_success
                        s[spf + "n_rollouts"] += rollout_stats.n_rollouts
                        s[spf + "success_rate"] = (
                            s[spf + "n_success"] / s[spf + "n_rollouts"]
                        )

                        s[spf + "total_return"] = (
                            s.get(spf + "total_return", 0) + rollout_stats.total_return
                        )
                        s[spf + "average_return"] = (
                            s[spf + "total_return"] / s[spf + "n_rollouts"]
                        )
                        s[spf + "total_reward"] = (
                            s.get(spf + "total_reward", 0) + rollout_stats.total_reward
                        )
                        s[spf + "average_reward"] = (
                            s[spf + "total_reward"] / s[spf + "n_rollouts"]
                        )
                    else:
                        raise ValueError(f"Invalid how_update: {how_update}")

                    # Set the currently_evaluating flag to false
                    run.config["currently_evaluating"] = False

                    # Update the run to save the summary fields
                    run.update()

                else:
                    print("Not writing to wandb")

                # If we prioritize the runs with the fewest rollouts, break after the first run
                # so that we can sort the runs according to the number of rollouts and evaluate them again
                if args.prioritize_fewest_rollouts:
                    break

            # If not in continuous mode, break
            if not args.continuous_mode:
                break

            # Sleep for the interval
            print(
                f"Sleeping for {args.continuous_interval} seconds before checking for new runs..."
            )
            time.sleep(args.continuous_interval)
    finally:
        # Unset the "currently_evaluating" flag
        if args.wandb:
            print("Exiting the evaluation loop")
            print("Unsetting the currently_evaluating flag")
            run.config["currently_evaluating"] = False
            run.update()
