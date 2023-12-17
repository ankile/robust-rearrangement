from tracemalloc import start
import numpy as np
import torch
import torch.nn as nn
import zarr
from src.data.normalizer import StateActionNormalizer, get_data_stats
from src.data.augmentation import ImageAugmentation, random_translate
import torchvision.transforms.functional as F

from ipdb import set_trace as bp


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class FurnitureImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: StateActionNormalizer,
        augment_image: bool = False,
        data_subset: int = None,
    ):
        # read from zarr dataset
        dataset = zarr.open(dataset_path, "r")

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends = dataset["episode_ends"][:data_subset]
        print(f"Loading dataset of {len(self.episode_ends)} episodes")
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            "robot_state": dataset["robot_state"][: self.episode_ends[-1]],
            "action": dataset["action"][: self.episode_ends[-1]],
        }
        print("Loaded robot_state and action")

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        normalized_train_data = dict()
        for key, data in train_data.items():
            normalized_train_data[key] = normalizer(
                torch.from_numpy(data), key, forward=True
            ).numpy()

        print("Done normalizing robot_state and action, starting on images")
        # int8, [0,255], (N,224,224,3)
        normalized_train_data["color_image1"] = dataset["color_image1"][
            : self.episode_ends[-1]
        ]
        print("Loaded color_image1")
        normalized_train_data["color_image2"] = dataset["color_image2"][
            : self.episode_ends[-1]
        ]
        print("Loaded color_image2")

        assert normalized_train_data["color_image1"].shape[1:] == (224, 224, 3)
        assert normalized_train_data["color_image2"].shape[1:] == (224, 224, 3)

        # Add image augmentation
        self.augment_image = augment_image

        self.indices = indices
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        # Add action and observation dimensions to the dataset
        self.action_dim = train_data["action"].shape[-1]
        self.robot_state_dim = train_data["robot_state"].shape[-1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["color_image1"] = nsample["color_image1"][: self.obs_horizon, :]
        nsample["color_image2"] = nsample["color_image2"][: self.obs_horizon, :]

        if self.augment_image:
            max_translation = 10
            nsample["color_image1"] = random_translate(
                nsample["color_image1"], max_translation
            )
            nsample["color_image2"] = random_translate(
                nsample["color_image2"], max_translation
            )

        nsample["robot_state"] = nsample["robot_state"][: self.obs_horizon, :]

        return nsample


class FurnitureFeatureDataset(torch.utils.data.Dataset):
    """
    This is the dataset used for precomputed image features.
    """

    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: StateActionNormalizer,
        normalize_features: bool = False,
        data_subset: int = None,
    ):
        # Read from zarr dataset
        self.dataset = zarr.open(dataset_path, "r")

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends = self.dataset["episode_ends"][:data_subset]
        print(f"Loading dataset of {len(self.episode_ends)} episodes")
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            "robot_state": self.dataset["robot_state"][: self.episode_ends[-1]],
            "action": self.dataset["action"][: self.episode_ends[-1]],
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        normalized_train_data = dict()

        # float32, (N, embed_dim)
        normalized_train_data["feature1"] = self.dataset["feature1"][
            : self.episode_ends[-1]
        ]
        normalized_train_data["feature2"] = self.dataset["feature2"][
            : self.episode_ends[-1]
        ]

        if normalize_features:
            for feature in ["feature1", "feature2"]:
                data = normalized_train_data[feature]
                stats = get_data_stats(data)
                normalizer.stats[feature] = nn.ParameterDict(
                    {
                        "min": nn.Parameter(
                            torch.from_numpy(stats["min"]), requires_grad=False
                        ),
                        "max": nn.Parameter(
                            torch.from_numpy(stats["max"]), requires_grad=False
                        ),
                    }
                )
                normalized_train_data[feature] = normalizer(
                    torch.from_numpy(data), feature, forward=True
                ).numpy()

        # compute statistics and normalized data to [-1,1]
        for key, data in train_data.items():
            normalized_train_data[key] = normalizer(
                torch.from_numpy(data), key, forward=True
            ).numpy()

        self.indices = indices
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        # Add action and observation dimensions to the dataset
        self.action_dim = train_data["action"].shape[-1]
        self.robot_state_dim = train_data["robot_state"].shape[-1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["feature1"] = nsample["feature1"][: self.obs_horizon, :]
        nsample["feature2"] = nsample["feature2"][: self.obs_horizon, :]
        nsample["robot_state"] = nsample["robot_state"][: self.obs_horizon, :]

        return nsample


class FurnitureQFeatureDataset(FurnitureFeatureDataset):
    def __init__(self, action_horizon: int, *args, **kwargs):
        super().__init__(*args, action_horizon=action_horizon, **kwargs)

        # Also add in rewards to the dataset
        self.reward_dim = 1
        self.normalized_train_data["reward"] = self.dataset["rewards"][
            : self.episode_ends[-1]
        ]

        self.action_horizon = action_horizon

    def __getitem__(self, idx):
        # Get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # Get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # Add the current observation to the input
        nsample["curr_obs"] = dict(
            feature1=nsample["feature1"][: self.obs_horizon, :],
            feature2=nsample["feature2"][: self.obs_horizon, :],
            robot_state=nsample["robot_state"][: self.obs_horizon, :],
        )

        # Add the next obs to the input
        # |0|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5| idx
        # |o|o|                             observations:       2
        # | |a|a|a|a|a|a|a|a|               actions executed:   8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        # | | |r|r|r|r|r|r|r|r|             rewards:   2
        # This is the observation that happens after the self.action_horizon actions have executed
        # Will start at `obs_horizon - 1 + action_horizon - (obs_horizon - 1)`
        # (which simplifies to `action_horizon`)
        # and end at `start + obs_horizon`
        start_idx = self.action_horizon
        end_idx = start_idx + self.obs_horizon
        nsample["next_obs"] = dict(
            feature1=nsample["feature1"][start_idx:end_idx, :],
            feature2=nsample["feature2"][start_idx:end_idx, :],
            robot_state=nsample["robot_state"][start_idx:end_idx, :],
        )

        # Add the reward to the input
        # What rewards should be counted? The rewards that happen after the first action is executed, up to the last action
        # We sum these into a single reward for the entire sequence
        nsample["reward"] = nsample["reward"][start_idx:end_idx].sum()

        # Delete the features and robot state from the root of the input
        nsample.pop("feature1")
        nsample.pop("feature2")
        nsample.pop("robot_state")

        return nsample
