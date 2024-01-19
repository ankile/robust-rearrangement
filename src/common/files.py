import os
from pathlib import Path
from typing import Literal, Union, List

from src.common.types import (
    EncoderName,
    TaskName,
    Environments,
    DemoSources,
    Randomness,
    DemoStatus,
)


def get_processed_path(
    obs_type: Literal["image", "feature"] = "image",
    encoder: Union[EncoderName, None] = None,
    environment: Union[List[Environments], Environments, None] = "sim",
    task: Union[List[TaskName], TaskName, None] = "one_leg",
    demo_source: Union[List[DemoSources], DemoSources, None] = "scripted",
    randomness: Union[List[Randomness], Randomness, None] = None,
    success_cond: Union[List[DemoStatus], DemoStatus] = "success",
):
    path = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed"

    # Image observations and procomputed features can not be mixed
    path /= obs_type

    # If we are using features, we need to specify the encoder
    # and the language condition
    if obs_type == "feature":
        assert encoder is not None, "Encoder must be specified for feature observations"
        path /= encoder

    # We can mix sim and real environments
    path /= "-".join(sorted(environment))

    # We can mix different tasks
    path /= "-".join(sorted(task))

    # We can mix different demo sources
    path /= "-".join(sorted(demo_source))

    # We can mix different randomness levels
    path /= "-".join(sorted(randomness))

    # We can mix success and failure conditions
    path /= "-".join(sorted(success_cond))

    # Set the file extension
    path = path.with_suffix(".zarr")

    # Raise an error if the path does not exist
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    return path


def add_glob_part(paths, part):
    if part is None:
        return [path / "*" for path in paths]
    elif isinstance(part, str):
        return [path / part for path in paths]
    elif isinstance(part, list):
        return [path / p for path in paths for p in part]
    else:
        raise ValueError(f"Invalid part: {part}")


def get_raw_paths(
    environment: Union[List[Environments], Environments, None] = "sim",
    task: List[TaskName] = ["square_table"],
    demo_source: List[Literal["scripted", "rollout", "teleop"]] = ["scripted"],
    randomness: List[Literal["low", "med", "high"]] = [],
    success_cond: List[Literal["success", "failure"]] = ["success"],
):
    path = Path(os.environ["DATA_DIR_RAW"]) / "raw"

    paths = [path]

    # We can mix sim and real environments
    paths = add_glob_part(paths, environment)

    # Set the file extension
    path = path.with_suffix(".pkl")

    # Raise an error if the path does not exist
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")

    return path


def trajectory_save_dir(
    environment: Environments,
    task: TaskName,
    demo_source: DemoSources,
    randomness: Randomness,
) -> Path:
    # Make the path to the directory
    path = (
        Path(os.environ["DATA_DIR_RAW"])
        / "raw"
        / environment
        / demo_source
        / task
        / randomness
    )

    # Make the directory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


if __name__ == "__main__":
    print(
        get_processed_path(
            obs_type="feature",
            encoder="resnet50",
            environment=["sim", "real"],
            # task=["square_table", "round_table"],
            demo_source=["scripted", "rollout"],
        )
    )
