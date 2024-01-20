from collections import deque
import torch
import torch.nn as nn
from src.dataset.normalizer import StateActionNormalizer

from ipdb import set_trace as bp  # noqa
from torchvision import transforms

resize_transform = transforms.Resize((224, 224))


# Update the PostInitCaller to be compatible
class PostInitCaller(type(torch.nn.Module)):
    def __call__(cls, *args, **kwargs):
        # print(f"{cls.__name__}.__call__({args}, {kwargs})")
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class Actor(torch.nn.Module, metaclass=PostInitCaller):
    obs_horizon: int
    action_horizon: int
    normalizer: StateActionNormalizer
    feature_noise: bool = False
    feature_dropout: bool = False
    feature_layernorm: bool = False
    encoding_dim: int

    def __post_init__(self, *args, **kwargs):
        if self.feature_dropout:
            self.dropout = nn.Dropout(p=self.feature_dropout)

        if self.feature_layernorm:
            self.layernorm1 = nn.LayerNorm(self.encoding_dim).to(self.device)
            self.layernorm2 = nn.LayerNorm(self.encoding_dim).to(self.device)

        self.print_model_params()

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params / 1_000_000:.2f}M")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params / 1_000_000:.2f}M parameters")

    def _normalized_obs(self, obs: deque, flatten: bool = True):
        """
        Normalize the observations

        Takes in a deque of observations and normalizes them
        And concatenates them into a single tensor of shape (n_envs, obs_horizon * obs_dim)
        """
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        B = nrobot_state.shape[0]

        # from furniture_bench.perception.image_utils import resize, resize_crop
        # Get size of the image
        img_size = obs[0]["color_image1"].shape[-3:]

        # Images come in as obs_horizon x (n_envs, 224, 224, 3) concatenate to (n_envs * obs_horizon, 224, 224, 3)
        img1 = torch.cat([o["color_image1"].unsqueeze(1) for o in obs], dim=1).reshape(
            B * self.obs_horizon, *img_size
        )
        img2 = torch.cat([o["color_image2"].unsqueeze(1) for o in obs], dim=1).reshape(
            B * self.obs_horizon, *img_size
        )

        # Resize the images to 224x224
        img1 = resize_transform(img1.transpose(1, 3)).transpose(1, 3)
        img2 = resize_transform(img2.transpose(1, 3)).transpose(1, 3)

        # Encode the images and reshape back to (B, obs_horizon, -1)
        feature1 = self.encoder1(img1).reshape(B, self.obs_horizon, -1)
        feature2 = self.encoder2(img2).reshape(B, self.obs_horizon, -1)

        # Apply the regularization to the features
        feature1, feature2 = self.regularize_features(feature1, feature2)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    def regularize_features(self, feature1, feature2):
        if self.feature_layernorm:
            feature1 = self.layernorm1(feature1)
            feature2 = self.layernorm2(feature2)

        if self.feature_dropout:
            feature1 = self.dropout(feature1)
            feature2 = self.dropout(feature2)

        if self.feature_noise:
            # Add noise to the features
            feature1 = feature1 + torch.randn_like(feature1) * self.feature_noise
            feature2 = feature2 + torch.randn_like(feature2) * self.feature_noise

        return feature1, feature2

    def _training_obs(self, batch, flatten: bool = True):
        # The robot state is already normalized in the dataset
        nrobot_state = batch["robot_state"]
        B = nrobot_state.shape[0]

        if self.observation_type == "image":
            # Convert images from (batch_size, obs_horizon, 224, 224, 3) -> (batch_size * obs_horizon, 224, 224, 3)
            # so that it's compatible with the encoder
            image1 = batch["color_image1"].reshape(B * self.obs_horizon, 224, 224, 3)
            image2 = batch["color_image2"].reshape(B * self.obs_horizon, 224, 224, 3)

            # Encode images and reshape back to (B, obs_horizon, encoding_dim)
            feature1 = self.encoder1(image1).reshape(B, self.obs_horizon, -1)
            feature2 = self.encoder2(image2).reshape(B, self.obs_horizon, -1)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
            nobs = nobs.flatten(start_dim=1) if flatten else nobs

        elif self.observation_type == "feature":
            # All observations already normalized in the dataset
            feature1 = batch["feature1"]
            feature2 = batch["feature2"]

            # Apply the regularization to the features
            feature1, feature2 = self.regularize_features(feature1, feature2)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
            if flatten:
                # (B, obs_horizon, obs_dim) --> (B, obs_horizon * obs_dim)
                nobs = nobs.flatten(start_dim=1)

        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        return nobs

    def train_mode(self):
        """
        Set models to train mode
        """
        self.train()

    def eval_mode(self):
        """
        Set models to eval mode
        """
        self.eval()

    def action(self, obs: deque) -> torch.Tensor:
        """
        Given a deque of observations, predict the action

        The action is predicted for the next step for all the environments (n_envs, action_dim)
        """
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def set_task(self, task):
        """
        Set the task for the actor
        """
        pass
