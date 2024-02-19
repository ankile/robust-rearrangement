import os
from pathlib import Path
from typing import Union, List
from glob import glob
from ipdb import set_trace as bp

from src.common.types import (
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
        raise ValueError(f"Invalid part: {parts}")


def get_processed_path(
    environment: Union[List[Environments], Environments, None] = "sim",
    task: Union[List[TaskName], TaskName, None] = "one_leg",
    demo_source: Union[List[DemoSources], DemoSources, None] = "scripted",
    randomness: Union[List[Randomness], Randomness, None] = None,
    demo_outcome: Union[List[DemoStatus], DemoStatus] = "success",
) -> Path:
    path = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed"

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


def get_processed_paths(
    environment: Union[List[Environments], Environments, None] = "sim",
    task: Union[List[TaskName], TaskName, None] = None,
    demo_source: Union[List[DemoSources], DemoSources, None] = None,
    randomness: Union[List[Randomness], Randomness, None] = None,
    demo_outcome: Union[List[DemoStatus], DemoStatus] = "success",
) -> Path:
    """
    Takes in a set of parameters and returns a list of paths to
    zarr files that should be combined into the final dataset.
    """

    path = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed"

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
    paths = [path.with_suffix(".zarr") for path in paths]

    # Use glob to find all the zarr paths
    paths = [Path(path) for p in paths for path in glob(str(p), recursive=True)]

    print("Found the following paths:")
    for p in paths:
        print("   ", p)

    return paths


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
    demo_source: List[DemoSources] = ["teleop"],
    randomness: List[Randomness] = ["low"],
    demo_outcome: List[DemoStatus] = ["success"],
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
    create: bool = True,
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

    if create:
        # Make the directory if it does not exist
        path.mkdir(parents=True, exist_ok=True)

    return path


if __name__ == "__main__":
    paths = get_processed_paths(
        environment="sim",
        task=None,
        demo_source=["scripted", "teleop"],
        randomness=None,
        demo_outcome="success",
    )

    print("Found these zarr files:")
    for path in paths:
        print("   ", path)
