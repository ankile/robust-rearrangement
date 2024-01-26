from pathlib import Path
from typing import List, Union

import numpy as np
import zarr
from tqdm import tqdm
from ipdb import set_trace as bp


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


# def combine_zarr_datasets(zarr_paths: List[Path], keys: List[str]):
#     # Initialize dictionary to hold total shapes
#     total_shapes = {key: 0 for key in keys}
#     last_episode_end = 0

#     # First pass to calculate total shapes
#     for path in zarr_paths:
#         dataset = zarr.open(path, mode="r")
#         for key in keys:
#             data_shape = dataset[key].shape[0]
#             if key == "episode_ends":
#                 total_shapes[key] += data_shape
#             else:
#                 total_shapes[key] += data_shape

#     # Preallocate numpy arrays
#     combined_data = {}
#     for path in zarr_paths:
#         dataset = zarr.open(path, mode="r")
#         for key in keys:
#             dtype = dataset[key].dtype
#             if key not in combined_data:
#                 if key == "episode_ends":
#                     combined_data[key] = np.zeros(total_shapes[key], dtype=dtype)
#                 else:
#                     # Assuming other arrays are 2D, adjust if not
#                     combined_data[key] = np.zeros(
#                         (total_shapes[key], *dataset[key].shape[1:]), dtype=dtype
#                     )

#     # Current indices for insertion into preallocated arrays
#     current_indices = {key: 0 for key in keys}

#     # Second pass to populate arrays
#     for path in tqdm(zarr_paths, desc="Loading zarr files"):
#         dataset = zarr.open(path, mode="r")

#         for key in tqdm(keys, desc="Loading data", position=1):
#             data = dataset[key][...]
#             data_length = data.shape[0]

#             if key == "episode_ends":
#                 if last_episode_end != 0:
#                     data += last_episode_end
#                 last_episode_end = data[-1]

#             combined_data[key][
#                 current_indices[key] : current_indices[key] + data_length
#             ] = data
#             current_indices[key] += data_length

#     return combined_data


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
    # Example usage
    zarr_paths = [
        "/data/scratch/ankile/furniture-data/processed/sim/round_table/scripted/low/success.zarr",
        "/data/scratch/ankile/furniture-data/processed/sim/round_table/scripted/med/success.zarr",
    ]

    keys = [
        "robot_state",
        "color_image1",
    ]

    combined_data = combine_zarr_datasets(zarr_paths, keys, max_episodes=None)

    print(
        combined_data["robot_state"].shape,
        combined_data["color_image1"].shape,
        combined_data["episode_ends"].shape,
        combined_data["episode_ends"][-1],
    )
