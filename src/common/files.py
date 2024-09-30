import os
from pathlib import Path
from typing import Union, List
from glob import glob
from ipdb import set_trace as bp

from src.common.types import (
    Controllers,
    Domains,
    TaskName,
    DemoSources,
    Randomness,
    DemoStatus,
)

SCAN_ASSET_ROOT = Path(__file__).parent.parent.absolute() / "real2sim/assets"
SCAN_ASSET_FB_ROOT = (
    Path(__file__).parent.parent.parent.absolute()
    / "furniture-bench/furniture_bench/assets_no_tags"
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
    controller: Union[List[Controllers], Controllers, None] = None,
    domain: Union[List[Domains], Domains, None] = "sim",
    task: Union[List[TaskName], TaskName, None] = "one_leg",
    demo_source: Union[List[DemoSources], DemoSources, None] = "scripted",
    randomness: Union[List[Randomness], Randomness, None] = None,
    demo_outcome: Union[List[DemoStatus], DemoStatus, None] = "success",
    suffix: Union[str, None] = None,
) -> Path:
    path = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed"

    # We can mix controllers
    path = add_subdir(path, controller)

    # We can mix sim and real environments
    path = add_subdir(path, domain)

    # We can mix tasks
    path = add_subdir(path, task)

    # We can mix demo sources
    path = add_subdir(path, demo_source)

    # We can mix randomness
    path = add_subdir(path, randomness)

    # We can mix demo outcomes
    path = add_subdir(path, demo_outcome)

    # We can mix suffixes
    if suffix is not None:
        path = add_subdir(path, suffix)

    # Set the file extension
    path = path.with_suffix(".zarr")

    return path


def get_processed_paths(
    controller: Union[List[Controllers], Controllers, None] = None,
    domain: Union[List[Domains], Domains, None] = "sim",
    task: Union[List[TaskName], TaskName, None] = None,
    demo_source: Union[List[DemoSources], DemoSources, None] = None,
    randomness: Union[List[Randomness], Randomness, None] = None,
    demo_outcome: Union[List[DemoStatus], DemoStatus] = "success",
    suffix: Union[str, None] = None,
) -> Path:
    """
    Takes in a set of parameters and returns a list of paths to
    zarr files that should be combined into the final dataset.

    The suffix parameter is used to choose any bespoke datasets that
    are not covered by the other parameters (e.g., diffik-produced data).
    """

    path = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed"

    paths = [path]

    # We can mix controllers
    paths = add_glob_part(paths, controller)

    # We can mix sim and real environments
    paths = add_glob_part(paths, domain)

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

    # Add the suffix pattern to all paths
    if suffix is not None:
        paths = add_glob_part(paths, suffix)

    # Add the extension pattern to all paths
    paths = [path.with_suffix(".zarr") for path in paths]

    # Use glob to find all the zarr paths
    paths = [Path(path) for p in paths for path in glob(str(p), recursive=True)]

    return paths


def path_override(
    paths: List[Path],
) -> List[Path]:

    root = Path(os.environ["DATA_DIR_PROCESSED"]) / "processed"
    paths = [root / path for path in paths]
    return paths


def add_glob_part(paths, part) -> List[Path]:
    if part is None:
        if paths[0].parts[-1] == "**":
            return paths
        return [path / "**" for path in paths]
    elif isinstance(part, str):
        return [path / part for path in paths]
    elif isinstance(part, list):
        # Recursively add each part
        ret = []

        for p in part:
            ret.extend(add_glob_part(paths, p))

        return ret
    else:
        raise ValueError(f"Invalid part: {part}")


def get_raw_paths(
    controller: Union[List[Controllers], Controllers, None] = None,
    domain: Union[List[Domains], Domains, None] = "sim",
    task: List[TaskName] = ["square_table"],
    demo_source: List[DemoSources] = ["teleop"],
    randomness: List[Randomness] = ["low"],
    demo_outcome: List[DemoStatus] = ["success"],
    suffix: Union[str, None] = None,
) -> List[Path]:
    """
    Takes in a set of parameters and returns a list of paths to
    pickle files that should be combined into the final dataset.

    The suffix parameter is used to choose any bespoke datasets that
    are not covered by the other parameters (e.g., diffik-produced data).
    """
    path = Path(os.environ["DATA_DIR_RAW"]) / "raw"

    paths = [path]

    # We can mix controllers
    paths = add_glob_part(paths, controller)

    # We can mix sim and real environments
    paths = add_glob_part(paths, domain)

    # Add the task pattern to all paths
    paths = add_glob_part(paths, task)

    # Add the demo source pattern to all paths
    paths = add_glob_part(paths, demo_source)

    # Add the randomness pattern to all paths
    paths = add_glob_part(paths, randomness)

    # Add the suffix pattern to all paths
    if suffix is not None:
        paths = add_glob_part(paths, suffix)

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
    controller: Controllers,
    domain: Domains,
    task: TaskName,
    demo_source: DemoSources,
    randomness: Randomness,
    perturb: bool = False,
    create: bool = True,
    suffix: str = "",
) -> Path:

    # Make the path to the directory
    path = (
        Path(os.environ["DATA_DIR_RAW"])
        / "raw"
        / controller
        / domain
        / task
        / demo_source
        / randomness
        / suffix
    )

    if create:
        # Make the directory if it does not exist
        path.mkdir(parents=True, exist_ok=True)

    return path


if __name__ == "__main__":
    paths = get_processed_paths(
        domain="real",
        task="place_shade",
        demo_source="teleop",
        randomness="low",
        demo_outcome="success",
    )

    print("Found these zarr files:")
    for path in paths:
        print("   ", path)

    paths = get_raw_paths(
        domain="real",
        task="place_shade",
        demo_source="teleop",
        randomness="low",
        demo_outcome="success",
    )
