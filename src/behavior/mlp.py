from omegaconf import OmegaConf
import torch
import torch.nn as nn
from typing import Union
from collections import deque
from ipdb import set_trace as bp  # noqa

from src.behavior.base import Actor
from src.models.mlp import MLP
from src.models import get_encoder
from src.common.control import RotationMode
from src.dataset.normalizer import Normalizer

from src.common.geometry import proprioceptive_quat_to_6d_rotation


class MLPActor(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        normalizer: Normalizer,
        config,
    ) -> None:
        super().__init__()
        self.device = device

        actor_cfg = config.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = (
            10 if config.control.act_rot_repr == RotationMode.rot_6d else 8
        )
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon
        self.observation_type = config.observation_type

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        # Regularization
        self.feature_noise = config.regularization.feature_noise
        self.feature_dropout = config.regularization.feature_dropout
        self.feature_layernorm = config.regularization.feature_layernorm
        self.state_noise = config.regularization.get("state_noise", False)

        # Convert the stats to tensors on the device
        self.normalizer = normalizer.to(device)

        encoder_kwargs = OmegaConf.to_container(config.vision_encoder, resolve=True)
        encoder_name = encoder_kwargs.model
        freeze_encoder = encoder_kwargs.freeze

        self.encoder1 = get_encoder(
            encoder_name,
            device=device,
            **encoder_kwargs,
        )
        self.encoder2 = (
            self.encoder1
            if freeze_encoder
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

        self.model = MLP(
            input_dim=self.timestep_obs_dim * self.obs_horizon,
            output_dim=self.action_dim * self.pred_horizon,
            hidden_dims=actor_cfg.hidden_dims,
            dropout=actor_cfg.dropout,
            residual=actor_cfg.residual,
        ).to(device)

        loss_fn_name = actor_cfg.loss_fn if hasattr(actor_cfg, "loss_fn") else "MSELoss"
        self.loss_fn = getattr(nn, loss_fn_name)()

    # === Inference ===
    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations
        nobs = self._normalized_obs(obs)

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            # Predict normalized action
            naction = self.model(nobs).reshape(
                nobs.shape[0], self.pred_horizon, self.action_dim
            )

            # unnormalize action
            # (B, pred_horizon, action_dim)
            action_pred = self.normalizer(naction, "action", forward=False)

            # Add the actions to the queue
            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            for i in range(start, end):
                self.actions.append(action_pred[:, i, :])

        # Return the first action in the queue
        return self.actions.popleft()

    # === Training ===
    def compute_loss(self, batch):
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=True)

        # Action already normalized in the dataset
        naction = batch["action"]

        # forward pass
        naction_pred = self.model(obs_cond).reshape(
            naction.shape[0], self.pred_horizon, self.action_dim
        )

        loss = self.loss_fn(naction_pred, naction)

        return loss


class MLPStateActor(nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device],
        normalizer: Normalizer,
        config,
    ) -> None:
        super().__init__()
        self.device = device

        actor_cfg = config.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = (
            10 if config.control.act_rot_repr == RotationMode.rot_6d else 8
        )
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon
        self.observation_type = config.observation_type

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        # Convert the stats to tensors on the device
        self.normalizer = normalizer.to(device)

        # Get the dimension of the parts poses
        self.parts_poses_dim = (
            35  # self.normalizer.stats["parts_poses"]["min"].shape[0]
        )

        self.timestep_obs_dim = config.robot_state_dim + self.parts_poses_dim

        self.model = MLP(
            input_dim=self.timestep_obs_dim * self.obs_horizon,
            output_dim=self.action_dim * self.pred_horizon,
            hidden_dims=actor_cfg.hidden_dims,
            dropout=actor_cfg.dropout,
            residual=actor_cfg.residual,
        ).to(device)

        loss_fn_name = actor_cfg.loss_fn if hasattr(actor_cfg, "loss_fn") else "MSELoss"
        self.loss_fn = getattr(nn, loss_fn_name)()

    # === Inference ===
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

        # Convert parts_poses from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        parts_poses = torch.cat([o["parts_poses"].unsqueeze(1) for o in obs], dim=1)

        # Normalize the parts_poses
        nparts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, nparts_poses], dim=-1)

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations
        nobs = self._normalized_obs(obs)

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            # Predict normalized action
            naction = self.model(nobs).reshape(
                nobs.shape[0], self.pred_horizon, self.action_dim
            )

            # unnormalize action
            # (B, pred_horizon, action_dim)
            action_pred = self.normalizer(naction, "action", forward=False)

            # Add the actions to the queue
            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            for i in range(start, end):
                self.actions.append(action_pred[:, i, :])

        # Return the first action in the queue
        return self.actions.popleft()

    # === Training ===
    def _training_obs(self, batch, flatten: bool = True):
        # The robot state is already normalized in the dataset
        nrobot_state = batch["robot_state"]

        # The parts poses are already normalized in the dataset
        nparts_poses = batch["parts_poses"]

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, nparts_poses], dim=-1)

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    def compute_loss(self, batch):
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=True)

        # Action already normalized in the dataset
        naction = batch["action"]

        # forward pass
        naction_pred = self.model(obs_cond).reshape(
            naction.shape[0], self.pred_horizon, self.action_dim
        )

        loss = self.loss_fn(naction_pred, naction)

        return loss

    def train_mode(self):
        """
        Set models to train mode
        """
        pass

    def eval_mode(self):
        """
        Set models to eval mode
        """
        pass

    def set_task(self, *args, **kwargs):
        """
        Set the task for the actor
        """
        pass
