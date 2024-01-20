import os
from pathlib import Path
from typing import Literal, Union, List
from glob import glob
from ipdb import set_trace as bp

from src.common.types import (
    EncoderName,
    TaskName,
    Environments,
    DemoSources,
    Randomness,
    DemoStatus,
)


def add_subdir(path: Path, parts: Union[List[str], str, None]) -> Path:
    if parts is None:
        return path
    elif isinstance(parts, str):
        return path / parts
    elif isinstance(parts, list):
        return path / "-".join(sorted(parts))
    else:
        raise ValueError(f"Invalid part: {part}")


def get_processed_path(
    obs_type: Literal["image", "feature"] = "image",
    encoder: Union[EncoderName, None] = None,
    environment: Union[List[Environments], Environments, None] = "sim",
    task: Union[List[TaskName], TaskName, None] = "one_leg",
    demo_source: Union[List[DemoSources], DemoSources, None] = "scripted",
    randomness: Union[List[Randomness], Randomness, None] = None,
    demo_outcome: Union[List[DemoStatus], DemoStatus] = "success",
) -> Path:
    path = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed"

    # Image observations and procomputed features can not be mixed
    path /= obs_type

    # If we are using features, we need to specify the encoder
    # and the language condition
    if obs_type == "feature":
        assert encoder is not None, "Encoder must be specified for feature observations"
        path /= encoder

    # We can mix sim and real environments
    path = add_subdir(path, environment)

    # We can mix tasks
    path = add_subdir(path, task)

    # We can mix demo sources
    path = add_subdir(path, demo_source)

    # We can mix randomness
    path = add_subdir(path, randomness)

    # We can mix demo outcomes
    path = add_subdir(path, demo_outcome)

    # Set the file extension
    path = path.with_suffix(".zarr")

    return path


def add_glob_part(paths, part) -> List[Path]:
    if part is None:
        if paths[0].parts[-1] == "**":
            return paths
        return [path / "**" for path in paths]
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
    demo_outcome: List[Literal["success", "failure"]] = ["success"],
) -> List[Path]:
    path = Path(os.environ["DATA_DIR_RAW"]) / "raw"

    paths = [path]

    # We can mix sim and real environments
    paths = add_glob_part(paths, environment)

    # Add the task pattern to all paths
    paths = add_glob_part(paths, task)

    # Add the demo source pattern to all paths
    paths = add_glob_part(paths, demo_source)

    # Add the randomness pattern to all paths
    paths = add_glob_part(paths, randomness)

    # Add the demo outcome pattern to all paths
    paths = add_glob_part(paths, demo_outcome)

    # Add ** if we are not using an explicit demo outcome
    if demo_outcome is None and paths[0].parts[-1] != "**":
        paths = add_glob_part(paths, "**")

    # Add the extension pattern to all paths
    paths = [path / "*.pkl*" for path in paths]

    print("Found the following paths:")
    for p in paths:
        print("   ", p)

    # Use glob to find all the pickle files
    pickle_paths = [Path(path) for p in paths for path in glob(str(p), recursive=True)]

    return pickle_paths


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
        / task
        / demo_source
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
