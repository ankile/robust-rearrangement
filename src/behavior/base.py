import random
from typing import Dict, Tuple, Union
from collections import deque
from omegaconf import DictConfig, OmegaConf
from src.models.utils import PrintParamCountMixin
from src.models.vib import VIB
from src.models.vision import VisionEncoder
import torch
import torch.nn as nn
from src.dataset.normalizer import LinearNormalizer
from src.models import get_encoder

from ipdb import set_trace as bp  # noqa
from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.common.vision import FrontCameraTransform, WristCameraTransform

import src.common.geometry as C


# Update the PostInitCaller to be compatible
class PostInitCaller(type(torch.nn.Module)):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class Actor(torch.nn.Module, PrintParamCountMixin, metaclass=PostInitCaller):
    obs_horizon: int
    action_horizon: int

    # Regularization
    feature_noise: bool = False
    feature_dropout: bool = False
    feature_layernorm: bool = False
    state_noise: float = 0.0
    proprioception_dropout: float = 0.0
    front_camera_dropout: float = 0.0
    wrist_camera_dropout: float = 0.0
    vib_front_feature_beta: float = 0.0
    confusion_loss_beta: float = 0.0
    confusion_loss_centroid_formulation: bool = False
    rescale_loss_for_domain: bool = False
    confusion_loss_anchored: bool = False
    weight_confusion_loss_by_action: bool = False

    encoding_dim: int
    augment_image: bool

    model: nn.Module

    camera1_transform = WristCameraTransform(mode="eval")
    camera2_transform = FrontCameraTransform(mode="eval")

    encoder1: VisionEncoder
    encoder1_proj: nn.Module

    encoder2: VisionEncoder
    encoder2_proj: nn.Module

    camera_2_vib: VIB

    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ):
        super().__init__()
        self.normalizer = LinearNormalizer()
        self.camera_2_vib = None

        actor_cfg = cfg.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = cfg.action_dim
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon
        self.predict_past_actions = actor_cfg.predict_past_actions

        # A queue of the next actions to be executed in the current horizon
        self.observations = deque(maxlen=self.obs_horizon)
        self.actions = deque(maxlen=self.action_horizon)

        self.observation_type = cfg.observation_type

        # Define what parts of the robot state to use
        self.include_proprioceptive_pos = actor_cfg.get(
            "include_proprioceptive_pos", True
        )
        self.include_proprioceptive_ori = actor_cfg.get(
            "include_proprioceptive_ori", True
        )

        # Regularization
        self.augment_image = cfg.data.augment_image
        self.confusion_loss_beta = actor_cfg.get("confusion_loss_beta", 0.0)

        self.device = device
        self.action_type = cfg.control.control_mode

        # Convert the stats to tensors on the device
        if self.observation_type == "image":
            self._initiate_image_encoder(cfg)

            self.feature_noise = cfg.regularization.get("feature_noise", None)
            self.feature_dropout = cfg.regularization.get("feature_dropout", None)
            self.feature_layernorm = cfg.regularization.get("feature_layernorm", None)
            self.state_noise = cfg.regularization.get("state_noise", 0.0)
            self.proprioception_dropout = cfg.regularization.get(
                "proprioception_dropout", 0.0
            )
            self.front_camera_dropout = cfg.regularization.get(
                "front_camera_dropout", 0.0
            )

            self.vib_front_feature_beta = cfg.regularization.get(
                "vib_front_feature_beta", 0.0
            )
            self.rescale_loss_for_domain = actor_cfg.get(
                "rescale_loss_for_domain", False
            )
            self.confusion_loss_anchored = actor_cfg.get(
                "confusion_loss_anchored", False
            )
            self.weight_confusion_loss_by_action = actor_cfg.get(
                "weight_confusion_loss_by_action", False
            )

        elif self.observation_type == "state":
            self.robot_state_dim = cfg.robot_state_dim
            self.parts_poses_dim = cfg.parts_poses_dim
            self.timestep_obs_dim = cfg.robot_state_dim + cfg.parts_poses_dim
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        self.flatten_obs = cfg.actor.flatten_obs
        self.obs_dim = (
            self.timestep_obs_dim * self.obs_horizon
            if self.flatten_obs
            else self.timestep_obs_dim
        )

        loss_fn_name = actor_cfg.loss_fn if hasattr(actor_cfg, "loss_fn") else "MSELoss"
        self.loss_fn = getattr(nn, loss_fn_name)(reduction="none")

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

    def actor_parameters(self):
        """
        Return the parameters of the actor by filtering out the encoder parameters
        """
        return [
            p
            for n, p in self.named_parameters()
            if not (n.startswith("encoder1.") or n.startswith("encoder2."))
        ]

    def encoder_parameters(self):
        """
        Return the parameters of the encoder
        """
        return [
            p
            for n, p in self.named_parameters()
            if (n.startswith("encoder1.") or n.startswith("encoder2."))
        ]

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

    def _initiate_image_encoder(self, cfg):
        # === Encoder ===
        encoder_kwargs = OmegaConf.to_container(cfg.vision_encoder, resolve=True)
        device = self.device
        actor_cfg = cfg.actor
        encoder_name = cfg.vision_encoder.model
        self.freeze_encoder = cfg.vision_encoder.freeze

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

        self.timestep_obs_dim = cfg.robot_state_dim + 2 * self.encoding_dim

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
        # TODO: Change this so the environment outputs 6D rotation instead when that's the chosen control mode
        if robot_state.shape[-1] == 14:
            robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        robot_state[..., :3] *= int(self.include_proprioceptive_pos)
        robot_state[..., 3:9] *= int(self.include_proprioceptive_ori)

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
            image1 = image1.permute(0, 3, 1, 2)
            image2 = image2.permute(0, 3, 1, 2)

            # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
            image1: torch.Tensor = self.camera1_transform(image1)
            image2: torch.Tensor = self.camera2_transform(image2)

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

            # Clamp the observation to be bounded to [-3, 3]
            nobs = torch.clamp(nobs, -3, 3)

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
        raise NotImplementedError

    def _sample_action_pred(self, nobs):
        # Predict normalized action
        # (B, candidates, pred_horizon, action_dim)
        naction = self._normalized_action(nobs)

        # unnormalize action
        # (B, pred_horizon, action_dim)
        action_pred = self.normalizer(naction, "action", forward=False)
        B = action_pred.shape[0]

        # These actions may be `pos`, `delta`, or `relative`, if relative, we make them absolute
        if self.action_type == "relative":
            # Need the current EE position in the robot frame unnormalized
            curr_pose = self.normalizer(
                nobs.view(B, self.obs_horizon, -1)[:, -1, :16],
                "robot_state",
                forward=False,
            )[:, :9]
            curr_pos = curr_pose[:, :3]
            curr_ori_6d = curr_pose[:, 3:9]
            action_pred[:, :, :3] += curr_pos[:, None, :]

            # Each action in the chunk will be relative to the current EE pose
            curr_ori_quat_xyzw = C.rotation_6d_to_quaternion_xyzw(curr_ori_6d)

            # Calculate the relative rot action
            action_quat_xyzw = C.rotation_6d_to_quaternion_xyzw(action_pred[:, :, 3:9])

            # Apply the relative quat on top of the current quat to get the absolute quat
            action_quat_xyzw = C.quaternion_multiply(
                curr_ori_quat_xyzw[:, None], action_quat_xyzw
            )

            # Convert the absolute quat to 6D rotation
            action_pred[:, :, 3:9] = C.quaternion_xyzw_to_rotation_6d(action_quat_xyzw)

        # Add the actions to the queue
        # only take action_horizon number of actions
        start = self.obs_horizon - 1 if self.predict_past_actions else 0
        end = start + self.action_horizon
        actions = deque()
        for i in range(start, end):
            actions.append(action_pred[:, i, :])

        return actions

    @torch.no_grad()
    def action_pred(self, batch):
        """
        Predict the action given the batch of observations
        """
        # Normalize observations
        nobs = self._training_obs(batch, flatten=self.flatten_obs)

        # Predict the action
        naction = self._normalized_action(nobs)

        # Unnormalize the action
        action = self.normalizer(naction, "action", forward=False)

        return action

    @torch.no_grad()
    def action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Given a deque of observations, predict the action

        The action is predicted for the next step for all the environments (n_envs, action_dim)

        This function must account for if we predict the past actions or not
        """

        # Append the new observation to the queue and ensure we fill it up to the horizon
        self.observations.append(obs)
        while len(self.observations) < self.obs_horizon:
            self.observations.append(obs)

        # Normalize observations
        nobs = self._normalized_obs(self.observations, flatten=self.flatten_obs)

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            self.actions = self._sample_action_pred(nobs)

        # Return the first action in the queue
        return self.actions.popleft()

    # @torch.compile
    def action_normalized(self, obs: Dict[str, torch.Tensor]):
        action = self.action(obs)
        return self.normalizer(action, "action", forward=True)

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
            rot_mat = C.rotation_6d_to_matrix(nrobot_state[:, :, 3:9])
            rot = C.matrix_to_euler_angles(rot_mat, "XYZ")

            # Add noise to the position with variance of of 1 cm
            pos = pos + torch.randn_like(pos) * 0.01

            # Sample random rotations in x, y, z Euler angles with variance of 0.1 rad
            d_rot = torch.randn_like(rot) * 0.1

            # Apply the noise rotation to the current rotation
            rot = C.matrix_to_rotation_6d(
                rot_mat @ C.euler_angles_to_matrix(d_rot, "XYZ")
            )

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

            # Images now have the channels first
            assert image1.shape[-3:] == (3, 240, 320)

            # Reshape the images to (B * obs_horizon, C, H, W) for the encoder
            image1 = image1.reshape(B * self.obs_horizon, *image1.shape[-3:])
            image2 = image2.reshape(B * self.obs_horizon, *image2.shape[-3:])

            # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
            # Since we're in training mode, the transform also performs augmentation
            image1: torch.Tensor = self.camera1_transform(image1)
            image2: torch.Tensor = self.camera2_transform(image2)

            # Encode images and reshape back to (B, obs_horizon, encoding_dim)
            feature1 = self.encoder1_proj(self.encoder1(image1)).reshape(
                B, self.obs_horizon, self.encoding_dim
            )
            feature2 = self.encoder2_proj(self.encoder2(image2)).reshape(
                B, self.obs_horizon, self.encoding_dim
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
            nobs = batch["obs"][:, : self.obs_horizon]

        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        # Take out the parts of the robot_state that we don't want
        nobs[..., :3] *= int(self.include_proprioceptive_pos)
        nobs[..., 3:9] *= int(self.include_proprioceptive_ori)

        if flatten:
            # (B, obs_horizon, obs_dim) --> (B, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        # Add a little bit of noise to the observations
        nobs = nobs + torch.randn_like(nobs) * self.state_noise

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

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.compute_loss(batch)

    def confusion_loss(self, batch, feature1, feature2):
        domain_idx: torch.Tensor = batch["domain"]

        # Split the embeddings into the two domains (sim/real)
        sim_emb1 = feature1[domain_idx == 0]  # N1 x 128
        real_emb1 = feature1[domain_idx == 1]  # N2 x 128

        sim_emb2 = feature2[domain_idx == 0]
        real_emb2 = feature2[domain_idx == 1]

        # Concatenate the embeddings along a new dimension
        sim_emb = torch.stack((sim_emb1, sim_emb2), dim=1)  # N1 x 2 x 128
        real_emb = torch.stack((real_emb1, real_emb2), dim=1)  # N2 x 2 x 128

        if self.confusion_loss_centroid_formulation:
            # Find the centroids of the embeddings
            sim_centroid = sim_emb.mean(dim=0)
            real_centroid = real_emb.mean(dim=0)

            # Compute the confusion loss using the centroids
            loss = torch.norm((real_centroid - sim_centroid), dim=-1).mean()

        else:
            # Randomly select the anchor domain (0 for sim, 1 for real)
            if self.confusion_loss_anchored:
                anchor_domain = random.randint(0, 1)
                if anchor_domain == 0:
                    # Use sim embeddings as the anchor
                    sim_emb = sim_emb.detach()
                else:
                    # Use real embeddings as the anchor
                    real_emb = real_emb.detach()

            sim_emb_expanded = sim_emb.unsqueeze(1)  # N1 x 1 x 2 x 128
            real_emb_expanded = real_emb.unsqueeze(0)  # 1 x N2 x 2 x 128

            # Compute the differences using broadcasting, N1 x N2 x 2
            differences = torch.norm((real_emb_expanded - sim_emb_expanded), dim=-1)

            if self.weight_confusion_loss_by_action:

                actions = batch["action"]  # Shape: (B, T, D)
                domain_idx = domain_idx.squeeze()  # Shape: (B,)

                # Split the actions into the two domains (sim/real)
                sim_actions = actions[domain_idx == 0]  # N1 x T x D
                real_actions = actions[domain_idx == 1]  # N2 x T x D

                # Compute the pairwise distances between actions
                sim_actions_expanded = sim_actions.unsqueeze(1)  # N1 x 1 x T x D
                real_actions_expanded = real_actions.unsqueeze(0)  # 1 x N2 x T x D

                # N1 x N2 x T
                action_distances = torch.norm(
                    (real_actions_expanded - sim_actions_expanded), dim=-1
                )

                # Compute the weights based on action distances
                weights = torch.exp(-action_distances)  # N1 x N2 x T
                weights = weights.mean(dim=-1)  # N1 x N2

                # Weight the differences using the computed weights
                differences = differences * weights.unsqueeze(-1)  # N1 x N2 x 2

            # Sum along all dimensions except the last to compute the accumulated loss
            loss = differences.mean(dim=(0, 1, 2))

        return loss

    def reset(self):
        """
        Reset the actor
        """
        self.actions.clear()
        self.observations.clear()

    # === Mode Toggle ===
    def train(self, mode=True):
        """
        Set models to train mode
        """
        super().train()
        if self.augment_image:
            self.camera1_transform.train()
            self.camera2_transform.train()
        else:
            self.camera1_transform.eval()
            self.camera2_transform.eval()

    def eval(self):
        """
        Set models to eval mode
        """
        super().train(mode=False)
        self.camera1_transform.eval()
        self.camera2_transform.eval()

        # Verify we are in eval mode
        for module in self.modules():
            assert not module.training

    def set_task(self, task):
        """
        Set the task for the actor
        """
        pass
