from pathlib import Path
import numpy as np
import torch
from typing import Union, List

from src.dataset.normalizer import Normalizer
from src.dataset.zarr import combine_zarr_datasets
from src.common.control import ControlMode

from src.common.tasks import furniture2idx
from src.common.vision import (
    FrontCameraTransform,
    WristCameraTransform,
)
from src.common.geometry import rot_6d_to_isaac_quat


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
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i]
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
        dataset_paths: Union[List[str], str],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: Normalizer,
        augment_image: bool = False,
        data_subset: int = None,
        first_action_idx: int = 0,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
    ):
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.control_mode = control_mode

        normalizer = normalizer.cpu()

        # Read from zarr dataset
        combined_data, metadata = combine_zarr_datasets(
            dataset_paths,
            [
                "color_image1",
                "color_image2",
                "robot_state",
                f"action/{control_mode}",
                "skill",
            ],
            max_episodes=data_subset,
            max_ep_cnt=max_episode_count,
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends = combined_data["episode_ends"]
        self.metadata = metadata
        print(f"Loading dataset of {len(self.episode_ends)} episodes:")
        for path, data in metadata.items():
            print(
                f"  {path}: {data['n_episodes_used']} episodes, {data['n_frames_used']}"
            )

        self.train_data = {
            "color_image1": combined_data["color_image1"],
            "color_image2": combined_data["color_image2"],
            "robot_state": combined_data["robot_state"],
            "action": combined_data[f"action/{control_mode}"],
        }

        # Normalize data to [-1,1]
        for key in normalizer.keys():
            self.train_data[key] = normalizer(self.train_data[key], key, forward=True)

        # compute start and end of each state-action sequence
        # also handles padding
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1 if pad_after else 0,
        )

        # Add image augmentation
        self.augment_image = augment_image
        self.image1_transform = WristCameraTransform(
            mode="train" if augment_image else "eval"
        )
        self.image2_transform = FrontCameraTransform(
            mode="train" if augment_image else "eval"
        )

        self.task_idxs = np.array(
            [furniture2idx[f] for f in combined_data["furniture"]]
        )
        self.successes = combined_data["success"].astype(np.uint8)
        self.skills = combined_data["skill"].astype(np.uint8)
        self.failure_idx = combined_data["failure_idx"]

        # Add action and observation dimensions to the dataset
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]

        # Take into account possibility of predicting an action that doesn't align with the first observation
        # TODO: Verify this works with the BC_RNN baseline
        self.first_action_idx = first_action_idx
        if first_action_idx < 0:
            self.first_action_idx = self.obs_horizon + first_action_idx

        self.final_action_idx = self.first_action_idx + self.pred_horizon

        if self.augment_image:
            self.train()
        else:
            self.eval()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
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
        nsample["robot_state"] = torch.from_numpy(
            nsample["robot_state"][: self.obs_horizon, :]
        )

        # Discard unused actions
        nsample["action"] = torch.from_numpy(
            nsample["action"][self.first_action_idx : self.final_action_idx, :]
        )

        # Apply the image augmentation
        nsample["color_image1"] = torch.stack(
            [
                self.image1_transform(img)
                for img in torch.from_numpy(nsample["color_image1"]).permute(0, 3, 1, 2)
            ]
        ).permute(0, 2, 3, 1)
        nsample["color_image2"] = torch.stack(
            [
                self.image2_transform(img)
                for img in torch.from_numpy(nsample["color_image2"]).permute(0, 3, 1, 2)
            ]
        ).permute(0, 2, 3, 1)

        # Add the task index and success flag to the sample
        nsample["task_idx"] = torch.LongTensor([self.task_idxs[demo_idx]])
        nsample["success"] = torch.IntTensor([self.successes[demo_idx]])

        return nsample

    def train(self):
        if self.augment_image:
            self.image1_transform.train()
            self.image2_transform.train()
        else:
            self.eval()

    def eval(self):
        self.image1_transform.eval()
        self.image2_transform.eval()


class FurnitureStateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths: Union[List[Path], Path],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: Union[Normalizer, None],
        data_subset: int = None,
        first_action_idx: int = 0,
        control_mode: ControlMode = ControlMode.delta,
        act_rot_repr: str = "quat",
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
    ):
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.control_mode = control_mode

        # Read from zarr dataset
        combined_data, metadata = combine_zarr_datasets(
            dataset_paths,
            [
                "parts_poses",
                "robot_state",
                # "color_image2",  # Debugging
                f"action/{control_mode}",
                "skill",
                "reward",
            ],
            max_episodes=data_subset,
            max_ep_cnt=max_episode_count,
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends: np.ndarray = combined_data["episode_ends"]
        self.metadata = metadata
        print(f"Loading dataset of {len(self.episode_ends)} episodes:")
        for path, data in metadata.items():
            print(
                f"  {path}: {data['n_episodes_used']} episodes, {data['n_frames_used']}"
            )

        # Convert robot_state orientation to quaternion
        robot_state = torch.from_numpy(combined_data["robot_state"])

        if act_rot_repr == "quat":
            robot_state = torch.cat(
                [
                    robot_state[:, :3],
                    rot_6d_to_isaac_quat(robot_state[:, 3:9]),
                    robot_state[:, 9:],
                ],
                dim=-1,
            )

        # Convert action orientation to quaternion
        action = torch.from_numpy(combined_data[f"action/{control_mode}"])

        if act_rot_repr == "quat":
            action = torch.cat(
                [
                    action[:, :3],
                    rot_6d_to_isaac_quat(action[:, 3:9]),
                    action[:, 9:],
                ],
                dim=-1,
            )

        self.train_data = {
            "robot_state": robot_state,
            "parts_poses": torch.from_numpy(combined_data["parts_poses"]),
            "action": action,
            # "color_image2": torch.from_numpy(combined_data["color_image2"]),  # Debugging
        }

        if normalizer is not None:
            normalizer.cpu()
            for key in self.train_data:
                if key in normalizer.keys():
                    continue
                normalizer.fit({key: self.train_data[key].numpy()})

            normalizer.cuda()

        if normalizer is not None:
            normalizer.cpu()
            for key in self.train_data:
                self.train_data[key] = normalizer(
                    self.train_data[key], key, forward=True
                )

            normalizer.cuda()

        # Recalculate the rewards and returns
        rewards = torch.zeros_like(self.train_data["parts_poses"][:, 0])
        rewards[self.episode_ends - 1] = 1.0

        gamma = 0.99
        returns = []
        ee = [0] + self.episode_ends.tolist()
        for start, end in zip(ee[:-1], ee[1:]):
            ep_rewards = rewards[start:end]
            timesteps = torch.arange(len(ep_rewards), device=ep_rewards.device)
            discounts = gamma**timesteps
            ep_returns = (
                torch.flip(
                    torch.cumsum(torch.flip(ep_rewards * discounts, dims=[0]), dim=0),
                    dims=[0],
                )
                / discounts
            )
            returns.append(ep_returns)

        # Concatenate the returns for all episodes into a single tensor
        returns = torch.cat(returns)
        self.train_data["returns"] = returns

        # compute start and end of each state-action sequence
        # also handles padding
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1 if pad_after else 0,
        )

        self.task_idxs = np.array(
            [furniture2idx[f] for f in combined_data["furniture"]]
        )
        self.successes = combined_data["success"].astype(np.uint8)
        self.skills = combined_data["skill"].astype(np.uint8)
        self.failure_idx = combined_data["failure_idx"]

        # Add action, robot_state, and parts_poses dimensions to the dataset
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]
        self.parts_poses_dim = self.train_data["parts_poses"].shape[-1]

        # Take into account possibility of predicting an action that doesn't align with the first observation
        # TODO: Verify this works with the BC_RNN baseline, or maybe just rip it out?
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
            demo_idx,
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

        # Discard unused actions
        nsample["action"] = nsample["action"][
            self.first_action_idx : self.final_action_idx, :
        ]

        # Discard unused observations
        robot_state = nsample["robot_state"][: self.obs_horizon, :]

        # Discard unused parts poses
        parts_poses = nsample["parts_poses"][: self.obs_horizon, :]

        nsample["obs"] = torch.cat([robot_state, parts_poses], dim=-1)

        # Discard unused returns
        nsample["returns"] = nsample["returns"][self.final_action_idx - 1]

        # # Add the task index and success flag to the sample
        # nsample["task_idx"] = torch.LongTensor([self.task_idxs[demo_idx]])
        # nsample["success"] = torch.IntTensor([self.successes[demo_idx]])

        return nsample

    def train(self):
        pass

    def eval(self):
        pass


class FurnitureStateTabletopDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths: Union[List[Path], Path],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: Union[Normalizer, None],
        data_subset: int = None,
        first_action_idx: int = 0,
        control_mode: ControlMode = ControlMode.delta,
        act_rot_repr: str = "quat",
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
    ):
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.control_mode = control_mode

        # Read from zarr dataset
        combined_data, metadata = combine_zarr_datasets(
            dataset_paths,
            [
                "parts_poses",
                "robot_state",
                # "color_image2",  # Debugging
                f"action/{control_mode}",
                "skill",
                "reward",
            ],
            max_episodes=data_subset,
            max_ep_cnt=max_episode_count,
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends: np.ndarray = combined_data["episode_ends"]
        self.metadata = metadata
        print(f"Loading dataset of {len(self.episode_ends)} episodes:")
        for path, data in metadata.items():
            print(
                f"  {path}: {data['n_episodes_used']} episodes, {data['n_frames_used']}"
            )

        # Convert robot_state orientation to quaternion
        robot_state = torch.from_numpy(combined_data["robot_state"])

        if act_rot_repr == "quat":
            robot_state = torch.cat(
                [
                    robot_state[:, :3],
                    rot_6d_to_isaac_quat(robot_state[:, 3:9]),
                    robot_state[:, 9:],
                ],
                dim=-1,
            )

        # Convert action orientation to quaternion
        action = torch.from_numpy(combined_data[f"action/{control_mode}"])

        if act_rot_repr == "quat":
            action = torch.cat(
                [
                    action[:, :3],
                    rot_6d_to_isaac_quat(action[:, 3:9]),
                    action[:, 9:],
                ],
                dim=-1,
            )

        self.train_data = {
            "robot_state": robot_state,
            "parts_poses": torch.from_numpy(combined_data["parts_poses"]),
            "action": action,
            # "color_image2": torch.from_numpy(combined_data["color_image2"]),  # Debugging
        }

        if normalizer is not None:
            normalizer.cpu()
            for key in self.train_data:
                if key in normalizer.keys():
                    continue
                normalizer.fit({key: self.train_data[key].numpy()})

            normalizer.cuda()

        # print("[NB] The actions are stored with rotations in 6D format")
        # Trim the data so that it's only up until the task is completed, i.e., the tabletop
        # is in location (0.0819, 0.2866, -0.0157)

        ee = np.array([0] + self.episode_ends.tolist())
        tabletop_goal = torch.tensor([0.0819, 0.2866, -0.0157])
        new_episode_starts = []
        new_episode_ends = []
        curr_cumulate_timesteps = 0
        self.episode_ends = []
        for prev_ee, curr_ee in zip(ee[:-1], ee[1:]):
            # Find the first index at which the tabletop goal is reached (if at all)
            for i in range(prev_ee, curr_ee):
                if torch.allclose(
                    self.train_data["parts_poses"][i, :3], tabletop_goal, atol=1e-2
                ):
                    new_episode_starts.append(prev_ee)
                    end = i + 10
                    new_episode_ends.append(end)
                    curr_cumulate_timesteps += end - prev_ee
                    self.episode_ends.append(curr_cumulate_timesteps)
                    break

        # Slice the train_data using the new episode starts and ends
        for key in self.train_data:
            data_slices = [
                self.train_data[key][start:end]
                for start, end in zip(new_episode_starts, new_episode_ends)
            ]
            self.train_data[key] = torch.cat(data_slices)

        self.episode_ends = torch.tensor(self.episode_ends)

        if normalizer is not None:
            normalizer.cpu()
            for key in self.train_data:
                self.train_data[key] = normalizer(
                    self.train_data[key], key, forward=True
                )

            normalizer.cuda()

        # Recalculate the rewards and returns
        rewards = torch.zeros_like(self.train_data["parts_poses"][:, 0])
        rewards[self.episode_ends - 1] = 1.0

        gamma = 0.99
        returns = []
        ee = [0] + self.episode_ends.tolist()
        for start, end in zip(ee[:-1], ee[1:]):
            ep_rewards = rewards[start:end]
            timesteps = torch.arange(len(ep_rewards), device=ep_rewards.device)
            discounts = gamma**timesteps
            ep_returns = (
                torch.flip(
                    torch.cumsum(torch.flip(ep_rewards * discounts, dims=[0]), dim=0),
                    dims=[0],
                )
                / discounts
            )
            returns.append(ep_returns)

        # Concatenate the returns for all episodes into a single tensor
        returns = torch.cat(returns)
        self.train_data["returns"] = returns

        # compute start and end of each state-action sequence
        # also handles padding
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1 if pad_after else 0,
        )

        self.task_idxs = np.array(
            [furniture2idx[f] for f in combined_data["furniture"]]
        )
        self.successes = combined_data["success"].astype(np.uint8)
        self.skills = combined_data["skill"].astype(np.uint8)
        self.failure_idx = combined_data["failure_idx"]

        # Add action, robot_state, and parts_poses dimensions to the dataset
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]
        self.parts_poses_dim = self.train_data["parts_poses"].shape[-1]

        # Take into account possibility of predicting an action that doesn't align with the first observation
        # TODO: Verify this works with the BC_RNN baseline, or maybe just rip it out?
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
            demo_idx,
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

        # Discard unused actions
        nsample["action"] = nsample["action"][
            self.first_action_idx : self.final_action_idx, :
        ]

        # Discard unused observations
        robot_state = nsample["robot_state"][: self.obs_horizon, :]

        # Discard unused parts poses
        parts_poses = nsample["parts_poses"][: self.obs_horizon, :]

        nsample["obs"] = torch.cat([robot_state, parts_poses], dim=-1)

        # Discard unused returns
        nsample["returns"] = nsample["returns"][self.final_action_idx - 1]

        # # Add the task index and success flag to the sample
        # nsample["task_idx"] = torch.LongTensor([self.task_idxs[demo_idx]])
        # nsample["success"] = torch.IntTensor([self.successes[demo_idx]])

        return nsample

    def train(self):
        pass

    def eval(self):
        pass
