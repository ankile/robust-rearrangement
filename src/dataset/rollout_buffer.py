from typing import Dict, List
import numpy as np
import torch
import gc


class RolloutBuffer:
    def __init__(
        self,
        max_size: int,
        state_dim: int,
        action_dim: int,
        pred_horizon: int = 8,
        obs_horizon: int = 1,
        action_horizon: int = 32,
        device: str = "cuda:0",
        predict_past_actions: bool = False,
        include_future_obs: bool = False,
        include_images: bool = False,
    ):

        self.device = device

        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        self.sequence_length = (
            pred_horizon if predict_past_actions else obs_horizon + pred_horizon - 1
        )

        # Set the limits for the action indices based on wether we predict past actions or not
        # First action refers to the first action we predict, not necessarily the first action executed
        self.first_action_idx = 0 if predict_past_actions else self.obs_horizon - 1
        self.final_action_idx = self.first_action_idx + self.pred_horizon
        self.last_obs = (
            self.obs_horizon if not include_future_obs else self.sequence_length
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(max_size, dtype=torch.float32)
        self.dones = torch.zeros(max_size, dtype=torch.bool)

        self.train_data = {
            "action": self.actions,
        }

        self.include_images = include_images
        if self.include_images:
            self.robot_states = torch.zeros((max_size, 16), dtype=torch.float32)
            self.color_image1 = torch.zeros((max_size, 240, 320, 3), dtype=torch.uint8)
            self.color_image2 = torch.zeros((max_size, 240, 320, 3), dtype=torch.uint8)
            self.train_data["robot_state"] = self.robot_states
            self.train_data["color_image1"] = self.color_image1
            self.train_data["color_image2"] = self.color_image2
        else:
            self.train_data["obs"] = self.states

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.indices = None

    @property
    def episode_end_idxs(self):
        return torch.where(self.dones[: self.size])[0].cpu().numpy() + 1

    @property
    def n_trajectories(self):
        return len(self.episode_end_idxs)

    def create_sample_indices(
        self,
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
                    [
                        buffer_start_idx,
                        buffer_end_idx,
                        sample_start_idx,
                        sample_end_idx,
                        i,
                    ]
                )
        indices = np.array(indices)
        return indices

    def sample_sequence(
        self,
        train_data: Dict[str, torch.Tensor],
        sequence_length: int,
        buffer_start_idx: int,
        buffer_end_idx: int,
        sample_start_idx: int,
        sample_end_idx: int,
    ) -> Dict[str, torch.Tensor]:
        result = dict()
        # TODO: Implement the performance improvement (particularly for image-based training):
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/sampler.py#L130-L138
        for key, input_arr in train_data.items():
            sample = input_arr[buffer_start_idx:buffer_end_idx]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
                data = torch.zeros(
                    size=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
                )
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

    def add_trajectories(
        self,
        *,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        states: torch.Tensor = None,
        robot_states: torch.Tensor = None,
        color_images1: torch.Tensor = None,
        color_images2: torch.Tensor = None,
    ):
        # Get the indices corresponding to the end of each episode with end index inclusive
        episode_ends = torch.where(dones)[0] + 1
        episode_idxs = torch.where(dones)[1]

        # Only add the timesteps that are part of the episode
        for ep_idx, ep_len in zip(episode_idxs, episode_ends):
            # print(f'Index: {ep_idx}, End: {end_idx}')
            # Decide what slice of the buffer to use - if the episode is too long, just cut it off
            restart = False
            if self.ptr + ep_len > self.max_size:
                # If the episode is too long, cut it off
                ep_len = self.max_size - self.ptr
                restart = True

            # Add the data to the buffer
            self.actions[self.ptr : self.ptr + ep_len] = actions[:ep_len, ep_idx]
            self.rewards[self.ptr : self.ptr + ep_len] = rewards[:ep_len, ep_idx]
            self.dones[self.ptr : self.ptr + ep_len] = dones[:ep_len, ep_idx]

            if self.include_images:
                assert robot_states is not None
                assert color_images1 is not None
                assert color_images2 is not None
                self.robot_states[self.ptr : self.ptr + ep_len] = robot_states[
                    :ep_len, ep_idx
                ]
                self.color_image1[self.ptr : self.ptr + ep_len] = color_images1[
                    :ep_len, ep_idx
                ]
                self.color_image2[self.ptr : self.ptr + ep_len] = color_images2[
                    :ep_len, ep_idx
                ]
            else:
                assert states is not None
                self.states[self.ptr : self.ptr + ep_len] = states[:ep_len, ep_idx]

            # Increment the start_idx (go to the next full episode)
            self.ptr = self.ptr + ep_len if not restart else 0
            self.size = min(self.size + ep_len, self.max_size)

    def form_batch(
        self, nsample_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        out_batch = dict()
        for key in nsample_list[0].keys():
            out_batch[key] = torch.stack(
                [nsample[key] for nsample in nsample_list], dim=0
            )
        return out_batch

    def rebuild_seq_indices(self):
        # First, get the valid indices depending on our episode ends and sequence length
        # episode_ends = torch.where(self.dones[: self.size])[0].cpu().numpy()
        # This expects the episode_ends to be the last index of the episode, not inclusive
        self.indices = self.create_sample_indices(
            self.episode_end_idxs,
            sequence_length=self.sequence_length,
            pad_before=self.obs_horizon - 1,
            pad_after=self.action_horizon - 1,
        )

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
        nsample = self.sample_sequence(
            train_data=self.train_data,
            sequence_length=self.sequence_length,
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
        if self.include_images:
            nsample["robot_state"] = nsample["robot_state"][: self.last_obs, :]
            nsample["color_image1"] = nsample["color_image1"][: self.last_obs, :]
            nsample["color_image2"] = nsample["color_image2"][: self.last_obs, :]

            nsample["color_image1"] = nsample["color_image1"].permute(0, 3, 1, 2)
            nsample["color_image2"] = nsample["color_image2"].permute(0, 3, 1, 2)
        else:
            nsample["obs"] = nsample["obs"][: self.last_obs, :]

        return nsample

    def __len__(self):
        return len(self.indices)
