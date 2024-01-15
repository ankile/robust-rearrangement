import numpy as np
import torch
import zarr
from src.dataset.normalizer import StateActionNormalizer
from src.dataset.augmentation import ImageAugmentation
from src.dataset.zarr_mod import ZarrSubsetView
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
    image_augmentation = None

    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: StateActionNormalizer,
        augment_image: bool = False,
        data_subset: int = None,
        first_action_idx: int = 0,
    ):
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.normalizer = normalizer.cpu()

        # Store the zarr dataset for later reading into
        # (actions and robot_state will not be normalized in batch)
        self.train_data = ZarrSubsetView(
            zarr_group=zarr.open(dataset_path, "r"),
            include_keys=["color_image1", "color_image2", "robot_state", "action"],
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends = self.train_data["episode_ends"][:data_subset]
        print(f"Loading dataset of {len(self.episode_ends)} episodes")

        # compute start and end of each state-action sequence
        # also handles padding
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # Add image augmentation
        self.augment_image = augment_image
        self.image_augmentation = ImageAugmentation(
            random_translate=True,
            color_jitter=True,
        )

        # Add action and observation dimensions to the dataset
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]

        # Take into account possibility of predicting an action that doesn't align with the first observation
        self.first_action_idx = first_action_idx
        if first_action_idx < 0:
            self.first_action_idx = self.obs_horizon + first_action_idx

        self.final_action_idx = self.first_action_idx + self.pred_horizon

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
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # Discard unused observations
        nsample["color_image1"] = nsample["color_image1"][: self.obs_horizon, :]
        nsample["color_image2"] = nsample["color_image2"][: self.obs_horizon, :]
        nsample["robot_state"] = nsample["robot_state"][: self.obs_horizon, :]

        # Discard unused actions
        nsample["action"] = nsample["action"][
            self.first_action_idx : self.final_action_idx, :
        ]

        # Normalize the robot state and actions
        for key in ["robot_state", "action"]:
            nsample[key] = self.normalizer(
                torch.from_numpy(nsample[key]), key, forward=True
            ).numpy()

        if self.augment_image:
            # Image augmentation function accepts one image at a time, need to loop over the batch
            nsample["color_image1"] = np.stack(
                [self.image_augmentation(sample) for sample in nsample["color_image1"]]
            )
            nsample["color_image2"] = np.stack(
                [self.image_augmentation(sample) for sample in nsample["color_image2"]]
            )

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
        data_subset: int = None,
        first_action_idx: int = 0,
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

        # Normalize data to [-1,1]
        for key, data in train_data.items():
            normalized_train_data[key] = normalizer(
                torch.from_numpy(data), key, forward=True
            ).numpy()

        self.indices = indices
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.feature_dim = normalized_train_data["feature1"].shape[-1]

        # Add action and observation dimensions to the dataset
        self.action_dim = train_data["action"].shape[-1]
        self.robot_state_dim = train_data["robot_state"].shape[-1]

        # Take into account possibility of predicting an action that doesn't align with the first observation
        self.first_action_idx = first_action_idx
        if first_action_idx < 0:
            self.first_action_idx = self.obs_horizon + first_action_idx

        # self.final_action_idx = self.first_action_idx + self.action_horizon
        self.final_action_idx = self.first_action_idx + self.pred_horizon

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

        # Discard unused actions
        nsample["action"] = nsample["action"][
            self.first_action_idx : self.final_action_idx, :
        ]

        # for diffusion policy version (self.first_action_offset = 0)
        # |0|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5| idx
        # |o|o|                             observations:       2
        # | |a|a|a|a|a|a|a|a|               actions executed:   8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        # for RNN version (self.first_action_offset = -1) -- meaning the first action aligns with the last observation
        # |0|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5| idx
        # |o|o|o|o|o|o|o|o|o|o|             observations:       2
        # |                 |p|             actions predicted:  1
        # |                 |a|             actions executed:   1

        return nsample


class OfflineRLFeatureDataset(FurnitureFeatureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Also add in rewards and terminal states to the dataset
        self.normalized_train_data["reward"] = self.dataset["reward"][
            : self.episode_ends[-1]
        ]
        self.normalized_train_data["terminal"] = self.dataset["terminal"][
            : self.episode_ends[-1]
        ]

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

        output = dict(
            action=nsample["action"],
            terminal=int(nsample["terminal"].sum() > 0),
        )

        # Add the current observation to the input
        output["curr_obs"] = dict(
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
        # | | |r|r|r|r|r|r|r|r|             rewards:            8
        # This is the observation that happens after the self.action_horizon actions have executed
        # Will start at `obs_horizon - 1 + action_horizon - (obs_horizon - 1)`
        # (which simplifies to `action_horizon`)
        # and end at `start + obs_horizon`
        start_idx = self.action_horizon
        end_idx = start_idx + self.obs_horizon
        output["next_obs"] = dict(
            feature1=nsample["feature1"][start_idx:end_idx, :],
            feature2=nsample["feature2"][start_idx:end_idx, :],
            robot_state=nsample["robot_state"][start_idx:end_idx, :],
        )

        # Add the reward to the input
        # What rewards should be counted? The rewards that happen after the first action is executed, up to the last action
        # We sum these into a single reward for the entire sequence
        output["reward"] = nsample["reward"][start_idx:end_idx].sum()

        return output
