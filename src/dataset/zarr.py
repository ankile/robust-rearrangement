from pathlib import Path
from typing import List, Union

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


def combine_zarr_datasets(zarr_paths: Union[List[str], str], keys, max_episodes=None):
    """
    Combine multiple zarr datasets into a single dataset.

    This function assume some keys are always present:
    - episode_ends: The end index of each episode.
    - task:         The task name for each episode.
    - success:      Whether the episode was successful.

    These are all of the same length, i.e., the number of episodes.
    """

    if isinstance(zarr_paths, str):
        zarr_paths = [zarr_paths]

    last_episode_end = 0
    n_episodes = 0
    batch_size = 1000
    total_frames = 0
    total_episodes = 0

    # First pass to calculate total shapes
    for path in zarr_paths:
        dataset = zarr.open(path, mode="r")
        total_frames += dataset["episode_ends"][:max_episodes][-1]
        total_episodes += len(dataset["episode_ends"][:max_episodes])

    combined_data = {
        "episode_ends": np.zeros(total_episodes, dtype=np.int64),
        "furniture": [],
        "success": np.zeros(total_episodes, dtype=np.uint8),
    }
    for key in keys:
        combined_data[key] = np.zeros(
            (total_frames,) + dataset[key].shape[1:], dtype=dataset[key].dtype
        )

    for path in tqdm(zarr_paths, desc="Loading zarr files"):
        dataset = zarr.open(path, mode="r")
        end_idxs = dataset["episode_ends"][:max_episodes]

        # Add the frame-based data
        for key in tqdm(keys, desc="Loading data", position=1, leave=False):
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

        # Add the episode-based data
        combined_data["episode_ends"][n_episodes : n_episodes + len(end_idxs)] = (
            end_idxs + last_episode_end
        )
        combined_data["furniture"].extend(dataset["furniture"][:max_episodes])
        combined_data["success"][n_episodes : n_episodes + len(end_idxs)] = dataset[
            "success"
        ][:max_episodes]

        # Upddate the counters
        last_episode_end += end_idxs[-1]
        n_episodes += len(end_idxs)

    return combined_data


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
