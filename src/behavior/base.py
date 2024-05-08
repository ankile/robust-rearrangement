from typing import Tuple, Union
from collections import deque
from omegaconf import DictConfig, OmegaConf
from src.common.control import RotationMode
from src.models.vib import VIB
from src.models.vision import VisionEncoder
import torch
import torch.nn as nn
from src.dataset.normalizer import LinearNormalizer
from src.models import get_encoder

from ipdb import set_trace as bp  # noqa
from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.common.vision import FrontCameraTransform, WristCameraTransform

from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    matrix_to_rotation_6d,
    euler_angles_to_matrix,
)


# Update the PostInitCaller to be compatible
class PostInitCaller(type(torch.nn.Module)):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class Actor(torch.nn.Module, metaclass=PostInitCaller):
    obs_horizon: int
    action_horizon: int

    # Regularization
    feature_noise: bool = False
    feature_dropout: bool = False
    feature_layernorm: bool = False
    state_noise: bool = False
    proprioception_dropout: float = 0.0
    front_camera_dropout: float = 0.0
    wrist_camera_dropout: float = 0.0
    vib_front_feature_beta: float = 0.0
    confusion_loss_beta: float = 0.0

    encoding_dim: int
    augment_image: bool = True

    camera1_transform = WristCameraTransform(mode="eval")
    camera2_transform = FrontCameraTransform(mode="eval")

    encoder1: VisionEncoder
    encoder1_proj: nn.Module

    encoder2: VisionEncoder
    encoder2_proj: nn.Module

    camera_2_vib: VIB = None

    def __init__(
        self,
        device: Union[str, torch.device],
        config: DictConfig,
    ):
        super().__init__()
        self.normalizer = LinearNormalizer()

        actor_cfg = config.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = (
            10 if config.control.act_rot_repr == RotationMode.rot_6d else 8
        )
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon
        self.predict_past_actions = actor_cfg.predict_past_actions

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        self.observation_type = config.observation_type

        # Regularization
        self.feature_noise = config.regularization.get("feature_noise", None)
        self.feature_dropout = config.regularization.get("feature_dropout", None)
        self.feature_layernorm = config.regularization.get("feature_layernorm", None)
        self.state_noise = config.regularization.get("state_noise", False)
        self.proprioception_dropout = config.regularization.get(
            "proprioception_dropout", 0.0
        )
        self.front_camera_dropout = config.regularization.get(
            "front_camera_dropout", 0.0
        )

        self.vib_front_feature_beta = config.regularization.get(
            "vib_front_feature_beta", 0.0
        )
        self.confusion_loss_beta = actor_cfg.get("confusion_loss_beta", 0.0)

        self.device = device

        # Convert the stats to tensors on the device
        if self.observation_type == "image":
            self._initiate_image_encoder(config)
        elif self.observation_type == "state":
            self.timestep_obs_dim = config.robot_state_dim + config.parts_poses_dim
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        self.flatten_obs = config.actor.flatten_obs
        self.obs_dim = (
            self.timestep_obs_dim * self.obs_horizon
            if self.flatten_obs
            else self.timestep_obs_dim
        )

        loss_fn_name = actor_cfg.loss_fn if hasattr(actor_cfg, "loss_fn") else "MSELoss"
        self.loss_fn = getattr(nn, loss_fn_name)()

    def __post_init__(self, *args, **kwargs):

        if self.observation_type == "image":
            assert self.encoder1 is not None, "encoder1 is not defined"
            assert self.encoder2 is not None, "encoder2 is not defined"

            if self.feature_dropout:
                self.dropout = nn.Dropout(p=self.feature_dropout)

            if self.feature_layernorm:
                self.layernorm1 = nn.LayerNorm(self.encoding_dim).to(self.device)
                self.layernorm2 = nn.LayerNorm(self.encoding_dim).to(self.device)

            if self.vib_front_feature_beta > 0:
                self.camera_2_vib = VIB(self.encoding_dim, self.encoding_dim)
                self.camera_2_vib.to(self.device)

        self.print_model_params()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def load_state_dict(self, state_dict):
        # Extract the normalizer state dict from the overall state dict
        normalizer_state_dict = {
            key[len("normalizer.") :]: value
            for key, value in state_dict.items()
            if key.startswith("normalizer.")
        }

        # Load the normalizer state dict
        self.normalizer.load_state_dict(normalizer_state_dict)

        # Load the rest of the state dict
        super().load_state_dict(state_dict)

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params / 1_000_000:.2f}M")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params / 1_000_000:.2f}M parameters")

    def _initiate_image_encoder(self, config):
        # === Encoder ===
        encoder_kwargs = OmegaConf.to_container(config.vision_encoder, resolve=True)
        device = self.device
        actor_cfg = config.actor
        encoder_name = config.vision_encoder.model
        self.freeze_encoder = config.vision_encoder.freeze

        self.encoder1 = get_encoder(
            encoder_name,
            device=device,
            **encoder_kwargs,
        )
        self.encoder2 = (
            self.encoder1
            if self.freeze_encoder
            else get_encoder(
                encoder_name,
                device=device,
                **encoder_kwargs,
            )
        )
        self.encoding_dim = self.encoder1.encoding_dim

        if actor_cfg.get("projection_dim") is not None:
            self.encoder1_proj = nn.Linear(
                self.encoding_dim, actor_cfg.projection_dim
            ).to(device)
            self.encoder2_proj = nn.Linear(
                self.encoding_dim, actor_cfg.projection_dim
            ).to(device)
            self.encoding_dim = actor_cfg.projection_dim
        else:
            self.encoder1_proj = nn.Identity()
            self.encoder2_proj = nn.Identity()

        self.timestep_obs_dim = config.robot_state_dim + 2 * self.encoding_dim

    # === Inference Observations ===
    def _normalized_obs(self, obs: deque, flatten: bool = True):
        """
        Normalize the observations

        Takes in a deque of observations and normalizes them
        And concatenates them into a single tensor of shape (n_envs, obs_horizon * obs_dim)
        """
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)

        # Convert the robot_state to use rot_6d instead of quaternion
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        # Normalize the robot_state
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        B = nrobot_state.shape[0]

        if self.observation_type == "image":

            # Get size of the image
            img_size = obs[0]["color_image1"].shape[-3:]

            # Images come in as obs_horizon x (n_envs, 224, 224, 3) concatenate to (n_envs * obs_horizon, 224, 224, 3)
            image1 = torch.cat(
                [o["color_image1"].unsqueeze(1) for o in obs], dim=1
            ).reshape(B * self.obs_horizon, *img_size)
            image2 = torch.cat(
                [o["color_image2"].unsqueeze(1) for o in obs], dim=1
            ).reshape(B * self.obs_horizon, *img_size)

            # Move the channel to the front (B * obs_horizon, H, W, C) -> (B * obs_horizon, C, H, W)
            # TODO: Remove this changing back and forth of channels first and last
            image1 = image1.permute(0, 3, 1, 2)
            image2 = image2.permute(0, 3, 1, 2)

            # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
            image1: torch.Tensor = self.camera1_transform(image1)
            image2: torch.Tensor = self.camera2_transform(image2)

            # Place the channel back to the end (B * obs_horizon, C, 224, 224) -> (B * obs_horizon, 224, 224, C)
            image1 = image1.permute(0, 2, 3, 1)
            image2 = image2.permute(0, 2, 3, 1)

            # Encode the images and reshape back to (B, obs_horizon, -1)
            feature1: torch.Tensor = self.encoder1_proj(self.encoder1(image1)).reshape(
                B, self.obs_horizon, -1
            )
            feature2: torch.Tensor = self.encoder2_proj(self.encoder2(image2)).reshape(
                B, self.obs_horizon, -1
            )

            # Apply the regularization to the features
            if self.feature_layernorm:
                feature1 = self.layernorm1(feature1)
                feature2 = self.layernorm2(feature2)

            if self.camera_2_vib is not None:
                # Apply the VIB to the front camera features
                feature2 = self.camera_2_vib(feature2)

            # Reshape concatenate the features
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
        elif self.observation_type == "state":
            # Convert parts_poses from obs_horizon x (n_envs, parts_poses_dim) -> (n_envs, obs_horizon, parts_poses_dim)
            parts_poses = torch.cat([o["parts_poses"].unsqueeze(1) for o in obs], dim=1)

            # Normalize the parts_poses
            nparts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

            # Concatenate the robot_state and parts_poses
            nobs = torch.cat([nrobot_state, nparts_poses], dim=-1)
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    # === Inference Actions ===
    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """
        Predict the normalized action given the normalized observations

        This is specific to the model and must be implemented by the subclass
        """

        if self.training and self.wrist_camera_dropout > 0:
            # print("[WARNING] Make sure this is disabled during evaluation")
            # Apply dropout to the front camera features, i.e., feature 2
            mask = (
                torch.rand(feature1.shape[0], self.obs_horizon, 1, device=self.device)
                > self.wrist_camera_dropout
            )
            feature1 = feature1 * mask

        if self.training and self.front_camera_dropout > 0:
            # print("[WARNING] Make sure this is disabled during evaluation")
            # Apply dropout to the front camera features, i.e., feature 2
            mask = (
                torch.rand(feature1.shape[0], self.obs_horizon, 1, device=self.device)
                > self.front_camera_dropout
            )
            feature2 = feature2 * mask

    def _sample_action_pred(self, nobs):
        # Predict normalized action
        # (B, candidates, pred_horizon, action_dim)
        naction = self._normalized_action(nobs)

        # unnormalize action
        # (B, pred_horizon, action_dim)
        action_pred = self.normalizer(naction, "action", forward=False)

        # Add the actions to the queue
        # only take action_horizon number of actions
        start = self.obs_horizon - 1 if self.predict_past_actions else 0
        end = start + self.action_horizon
        actions = deque()
        for i in range(start, end):
            actions.append(action_pred[:, i, :])

        return actions

    @torch.no_grad()
    def action(self, obs: deque):
        """
        Given a deque of observations, predict the action

        The action is predicted for the next step for all the environments (n_envs, action_dim)

        This function must account for if we predict the past actions or not
        """

        # Normalize observations
        nobs = self._normalized_obs(obs, flatten=self.flatten_obs)

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            self.actions = self._sample_action_pred(nobs)

        # Return the first action in the queue
        return self.actions.popleft()

    # === Training Observations ===
    def _training_obs(self, batch, flatten: bool = True):

        # Check if we're in training mode and we want to add noise to the robot state
        if self.training and self.state_noise:
            # The robot state is already normalized in the dataset
            nrobot_state = batch["robot_state"]
            B = nrobot_state.shape[0]

            # Add noise to the robot state akin to Ke et al., “Grasping with Chopsticks.”
            # Extract only the current position and orientation (x, y, x and 6D rotation)
            pos = nrobot_state[:, :, :3]
            rot_mat = rotation_6d_to_matrix(nrobot_state[:, :, 3:9])
            rot = matrix_to_euler_angles(rot_mat, "XYZ")

            # Add noise to the position with variance of of 1 cm
            pos = pos + torch.randn_like(pos) * 0.01

            # Sample random rotations in x, y, z Euler angles with variance of 0.1 rad
            d_rot = torch.randn_like(rot) * 0.1

            # Apply the noise rotation to the current rotation
            rot = matrix_to_rotation_6d(rot_mat @ euler_angles_to_matrix(d_rot, "XYZ"))

            # In 20% of observations, we now the noised position and rotation to the robot state
            mask = torch.rand(B) < 0.2
            nrobot_state[mask, :, :3] = pos[mask]
            nrobot_state[mask, :, 3:9] = rot[mask]

            batch["robot_state"] = nrobot_state

        if self.training and self.proprioception_dropout > 0:
            # The robot state is already normalized in the dataset
            nrobot_state = batch["robot_state"]
            B = nrobot_state.shape[0]

            # Apply dropout to the full robot state with probability of self.proprioception_dropout
            mask = (
                torch.rand(B, self.obs_horizon, 1, device=self.device)
                > self.proprioception_dropout
            )
            nrobot_state = nrobot_state * mask

            batch["robot_state"] = nrobot_state

        if self.observation_type == "image":
            # The robot state is already normalized in the dataset
            nrobot_state = batch["robot_state"]
            B = nrobot_state.shape[0]

            image1: torch.Tensor = batch["color_image1"]
            image2: torch.Tensor = batch["color_image2"]

            # TODO: Remove this changing back and forth of channels first and last
            # Reshape the images to (B * obs_horizon, H, W, C) for the encoder
            image1 = image1.reshape(B * self.obs_horizon, *image1.shape[-3:])
            image2 = image2.reshape(B * self.obs_horizon, *image2.shape[-3:])

            # Move the channel to the front (B * obs_horizon, H, W, C) -> (B * obs_horizon, C, H, W)
            image1 = image1.permute(0, 3, 1, 2)
            image2 = image2.permute(0, 3, 1, 2)

            # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
            # Since we're in training mode, the tranform also performs augmentation
            image1: torch.Tensor = self.camera1_transform(image1)
            image2: torch.Tensor = self.camera2_transform(image2)

            # Place the channel back to the end (B * obs_horizon, C, 224, 224) -> (B * obs_horizon, 224, 224, C)
            image1 = image1.permute(0, 2, 3, 1)
            image2 = image2.permute(0, 2, 3, 1)

            # Encode images and reshape back to (B, obs_horizon, encoding_dim)
            feature1 = self.encoder1_proj(self.encoder1(image1)).reshape(
                B, self.obs_horizon, -1
            )
            feature2 = self.encoder2_proj(self.encoder2(image2)).reshape(
                B, self.obs_horizon, -1
            )

            # Apply the regularization to the features
            feature1, feature2 = self.regularize_features(feature1, feature2)

            if self.feature_layernorm:
                feature1 = self.layernorm1(feature1)
                feature2 = self.layernorm2(feature2)

            if self.camera_2_vib is not None:
                feature2, mu, log_var = self.camera_2_vib.train_sample(feature2)

                # Store the mu and log_var for the loss computation later
                batch["mu"] = mu
                batch["log_var"] = log_var

            if self.confusion_loss_beta > 0:
                # Apply the confusion loss to the front camera features
                confusion_loss = self.confusion_loss(batch, feature1, feature2)
                batch["confusion_loss"] = confusion_loss

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)

        elif self.observation_type == "state":
            # Parts poses are already normalized in the dataset
            nobs = batch["obs"]

        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        if flatten:
            # (B, obs_horizon, obs_dim) --> (B, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    def regularize_features(
        self, feature1: torch.Tensor, feature2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.feature_dropout > 0:
            # print("[WARNING] Make sure this is disabled during evaluation")
            feature1 = self.dropout(feature1)
            feature2 = self.dropout(feature2)

        if self.training and self.feature_noise:
            # print("[WARNING] Make sure this is disabled during evaluation")
            # Add noise to the features
            feature1 = feature1 + torch.randn_like(feature1) * self.feature_noise
            feature2 = feature2 + torch.randn_like(feature2) * self.feature_noise

        if self.training and self.wrist_camera_dropout > 0:
            # print("[WARNING] Make sure this is disabled during evaluation")
            # Apply dropout to the front camera features, i.e., feature 2
            mask = (
                torch.rand(feature1.shape[0], self.obs_horizon, 1, device=self.device)
                > self.wrist_camera_dropout
            )
            feature1 = feature1 * mask

        if self.training and self.front_camera_dropout > 0:
            # print("[WARNING] Make sure this is disabled during evaluation")
            # Apply dropout to the front camera features, i.e., feature 2
            mask = (
                torch.rand(feature1.shape[0], self.obs_horizon, 1, device=self.device)
                > self.front_camera_dropout
            )
            feature2 = feature2 * mask

        return feature1, feature2

    def compute_loss(self, batch):
        raise NotImplementedError

    def confusion_loss(self, batch, feature1, feature2):
        domain_idx = batch["domain"]

        # Split the embeddings into the two domains (sim/real)
        sim_emb1 = feature1[domain_idx == 0]  # N1 x 128
        real_emb1 = feature1[domain_idx == 1]  # N2 x 128

        real_emb1_expanded = real_emb1.unsqueeze(1)
        sim_emb1_expanded = sim_emb1.unsqueeze(0)

        # Subtract using broadcasting, resulting shape is [N1, N2, 128]
        differences1 = torch.norm((real_emb1_expanded - sim_emb1_expanded), dim=-1)

        # Split the embeddings into the two domains (sim/real)
        sim_emb2 = feature2[domain_idx == 0]
        real_emb2 = feature2[domain_idx == 1]

        real_emb2_expanded = real_emb2.unsqueeze(1)
        sim_emb2_expanded = sim_emb2.unsqueeze(0)

        # Subtract using broadcasting, resulting shape is [N1, N2, 128]
        differences2 = torch.norm((real_emb2_expanded - sim_emb2_expanded), dim=-1)

        # Sum along all dimensions except the last to compute the accumulated loss
        # Final shape after sum will be [128], so another sum over the last dimension is needed
        loss = differences1.mean(dim=(0, 1)) + differences2.mean(dim=(0, 1))

        return loss

    # === Mode Toggle ===
    def train_mode(self):
        """
        Set models to train mode
        """
        self.train()
        if self.augment_image:
            self.camera1_transform.train()
            self.camera2_transform.train()
        else:
            self.camera1_transform.eval()
            self.camera2_transform.eval()

    def eval_mode(self):
        """
        Set models to eval mode
        """
        self.eval()
        self.camera2_transform.eval()

    def action(self, obs: deque) -> torch.Tensor:
        """
        Given a deque of observations, predict the action

        The action is predicted for the next step for all the environments (n_envs, action_dim)
        """
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def set_task(self, task):
        """
        Set the task for the actor
        """
        pass
