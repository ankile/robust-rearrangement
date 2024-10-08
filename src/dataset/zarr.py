from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import zarr
from tqdm import tqdm
from ipdb import set_trace as bp

from src.common.files import get_processed_paths


class ZarrSubsetView:
    def __init__(self, zarr_group, include_keys):
        """
        Create a view-like object for a Zarr group, excluding specified keys.
        :param zarr_group: The original Zarr group.
        :param exclude_keys: A set or list of keys to exclude.
        """
        self.zarr_group = zarr_group
        self.include_keys = set(include_keys)

    def __getitem__(self, key):
        return self.zarr_group[key]

    def observation_keys(self):
        """
        Return keys not excluded.
        """
        return [key for key in self.zarr_group.keys() if key in self.include_keys]

    def items(self):
        """
        Return items not excluded.
        """
        return [(key, self.zarr_group[key]) for key in self.observation_keys()]


def dataset_tuple(path: Path) -> Tuple[str, str, str, str]:
    """
    Extract the task, source, randomness, and outcome from a zarr path.
    """
    return path.with_name(path.stem).parts[-4:]


def combine_zarr_datasets(
    zarr_paths: Union[List[Path], Path],
    keys,
    max_episodes=None,
    max_ep_cnt=None,
) -> Tuple[dict, dict]:
    """
    Combine multiple zarr datasets into a single dataset.

    This function assume some keys are always present:
    - episode_ends: The end index of each episode.
    - task:         The task name for each episode.
    - success:      Whether the episode was successful.

    These are all of the same length, i.e., the number of episodes.
    """

    if not isinstance(zarr_paths, list):
        zarr_paths = [zarr_paths]

    last_episode_end = 0
    n_episodes = 0
    batch_size = 1000
    total_frames = 0
    total_episodes = 0

    metadata = {}

    domain_idx = dict(sim=0, real=1)

    # First pass to calculate total shapes
    for path in zarr_paths:
        # [F]urniture, [S]ource, [R]andomness, [O]utcome
        f, s, r, o = dataset_tuple(path)
        dataset = zarr.open(path, mode="r")

        if max_ep_cnt is not None:
            max_ep = max_ep_cnt.get(f, {}).get(s, {}).get(r, {}).get(o, max_episodes)
        else:
            max_ep = max_episodes

        n_frames_in_dataset = dataset["episode_ends"][:max_ep][-1]
        n_ep_in_dataset = len(dataset["episode_ends"][:max_ep])

        # Add the metadata
        metadata[str(dataset_tuple(path))] = {
            "n_episodes_used": n_ep_in_dataset,
            "n_frames_used": n_frames_in_dataset,
            "attrs": dataset.attrs.asdict(),
        }

        # Add the counts to the totals
        total_frames += n_frames_in_dataset
        total_episodes += n_ep_in_dataset

    combined_data = {
        "episode_ends": np.zeros(total_episodes, dtype=np.int64),
        "task": [],
        "success": np.zeros(total_episodes, dtype=np.uint8),
        # Domain is 0 for sim, 1 for real
        "domain": np.zeros(total_episodes, dtype=np.uint8),
        "zarr_idx": np.zeros(total_frames, dtype=np.uint8),
        "within_zarr_idx": np.zeros(total_frames, dtype=np.uint8),
    }
    for key in keys:
        combined_data[key] = np.zeros(
            (total_frames,) + dataset[key].shape[1:], dtype=dataset[key].dtype
        )

    for ii, path in enumerate(tqdm(zarr_paths, desc="Loading zarr files")):
        dataset = zarr.open(path, mode="r")
        # Get the max_episodes for this dataset
        max_episodes = metadata[str(dataset_tuple(path))]["n_episodes_used"]
        end_idxs = dataset["episode_ends"][:max_episodes]

        # Add the frame-based data
        for key in tqdm(keys, desc="Loading data", position=1, leave=False):

            # For the image data, we load in batches
            if key.startswith("color_image"):
                for i in tqdm(
                    range(0, end_idxs[-1], batch_size),
                    desc=f"Loading batches for {key}",
                    leave=False,
                    position=2,
                ):
                    end = min(i + batch_size, end_idxs[-1])
                    batch = dataset[key][i:end]
                    combined_data[key][
                        last_episode_end + i : last_episode_end + end
                    ] = batch

            # For the other data, we can load it all at once
            else:
                combined_data[key][
                    last_episode_end : last_episode_end + end_idxs[-1]
                ] = dataset[key][: end_idxs[-1]]

        # Add the episode-based data
        combined_data["episode_ends"][n_episodes : n_episodes + len(end_idxs)] = (
            end_idxs + last_episode_end
        )
        task = dataset.get("task", dataset.get("furniture"))
        combined_data["task"].extend(task[:max_episodes])
        combined_data["success"][n_episodes : n_episodes + len(end_idxs)] = dataset[
            "success"
        ][:max_episodes]

        combined_data["failure_idx"] = dataset.get(
            "failure_idx", np.full_like(end_idxs, -1)
        )
        combined_data["domain"][n_episodes : n_episodes + len(end_idxs)] = domain_idx[
            dataset.attrs["domain"][:max_episodes]
        ]

        combined_data["zarr_idx"][
            last_episode_end : last_episode_end + end_idxs[-1]
        ] = ii
        combined_data["within_zarr_idx"][
            last_episode_end : last_episode_end + end_idxs[-1]
        ] = np.arange(0, end_idxs[-1])

        # Upddate the counters
        last_episode_end += end_idxs[-1]
        n_episodes += len(end_idxs)

    return combined_data, metadata


if __name__ == "__main__":
    zarr_paths = get_processed_paths(
        environment="sim",
        task=None,
        demo_source=["scripted", "teleop"],
        randomness=None,
        demo_outcome="success",
    )
    print(len(zarr_paths))

    keys = [
        "color_image1",
        "color_image2",
        "robot_state",
        "action/delta",
    ]

    combined_data = combine_zarr_datasets(zarr_paths, keys, max_episodes=None)

    print(
        combined_data["robot_state"].shape,
        combined_data["color_image1"].shape,
        combined_data["episode_ends"].shape,
        combined_data["episode_ends"][-1],
    )
