from abc import ABC
from collections import deque
import torch
import torch.nn as nn
from src.data.normalizer import StateActionNormalizer
from src.models.vision import get_encoder
from src.models.unet import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.models.value import DoubleCritic, ValueNetwork

from ipdb import set_trace as bp  # noqa


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call BaseClass()"""
        print(f"{__class__.__name__}.__call__({args}, {kwargs})")
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class Actor(ABC, torch.nn.Module, metaclass=PostInitCaller):
    obs_horizon: int
    action_horizon: int

    def _normalized_obs(self, obs: deque):
        """
        Normalize the observations

        Takes in a deque of observations and normalizes them
        And concatenates them into a single tensor of shape (n_envs, obs_horizon * obs_dim)
        """
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        B = nrobot_state.shape[0]

        # Get size of the image
        img_size = obs[0]["color_image1"].shape[-3:]

        # Images come in as obs_horizon x (n_envs, 224, 224, 3) concatenate to (n_envs * obs_horizon, 224, 224, 3)
        img1 = torch.cat([o["color_image1"].unsqueeze(1) for o in obs], dim=1).reshape(
            B * self.obs_horizon, *img_size
        )
        img2 = torch.cat([o["color_image2"].unsqueeze(1) for o in obs], dim=1).reshape(
            B * self.obs_horizon, *img_size
        )

        # Encode the images and reshape back to (B, obs_horizon, -1)
        features1 = self.encoder1(img1).reshape(B, self.obs_horizon, -1)
        features2 = self.encoder2(img2).reshape(B, self.obs_horizon, -1)

        if "feature1" in self.normalizer.stats:
            features1 = self.normalizer(features1, "feature1", forward=True)
            features2 = self.normalizer(features2, "feature2", forward=True)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, features1, features2], dim=-1)
        # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
        nobs = nobs.flatten(start_dim=1)

        return nobs

    def _training_obs(self, batch):
        nrobot_state = batch["robot_state"]
        B = nrobot_state.shape[0]

        if self.observation_type == "image":
            # Convert images from obs_horizon x (n_envs, 224, 224, 3) -> (n_envs, obs_horizon, 224, 224, 3)
            # so that it's compatible with the encoder
            image1 = batch["color_image1"].reshape(B * self.obs_horizon, 224, 224, 3)
            image2 = batch["color_image2"].reshape(B * self.obs_horizon, 224, 224, 3)

            # Encode images and reshape back to (B, obs_horizon, -1)
            image1 = self.encoder1(image1).reshape(B, self.obs_horizon, -1)
            image2 = self.encoder2(image2).reshape(B, self.obs_horizon, -1)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, image1, image2], dim=-1)
            nobs = nobs.flatten(start_dim=1)

        elif self.observation_type == "feature":
            # All observations already normalized in the dataset
            feature1 = batch["feature1"]
            feature2 = batch["feature2"]

            if self.noise_augment:
                feature1 += torch.normal(
                    mean=0.0, std=self.noise_augment, size=feature1.size()
                ).to(feature1.device)
                feature2 += torch.normal(
                    mean=0.0, std=self.noise_augment, size=feature2.size()
                ).to(feature2.device)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
            nobs = nobs.flatten(start_dim=1)
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        return nobs

    def action(self, obs: deque) -> torch.Tensor:
        """
        Given a deque of observations, predict the action

        The action is predicted for the next step for all the environments (n_envs, action_dim)
        """
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def __post_init__(self, *args, **kwargs):
        raise NotImplementedError
