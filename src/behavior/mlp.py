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


import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SmallAgent(nn.Module):
    normalizer: Normalizer

    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=1),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, 1, np.prod(action_shape)) * 0)

    def training_obs(self, batch: dict, flatten: bool = True):
        # The robot state is already normalized in the dataset
        robot_state = batch["robot_state"]
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        # The parts poses are already normalized in the dataset
        parts_poses = batch["parts_poses"]
        nparts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, nparts_poses], dim=-1)

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

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
        naction_pred = self.actor_mean(obs_cond).reshape(
            naction.shape[0], self.pred_horizon, self.action_dim
        )

        loss = torch.nn.functional.mse_loss(naction_pred, naction)

        return loss

    def get_value(self, nobs: torch.Tensor):
        return self.critic(nobs)

    def get_action_and_value(self, nobs: torch.Tensor, action=None):
        action_mean = self.actor_mean(nobs).reshape(
            nobs.shape[0], self.pred_horizon, self.action_dim
        )
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            naction = probs.sample()
        else:
            naction = self.normalizer(action, "action", forward=True)

        action = self.normalizer(naction, "action", forward=False)
        return (
            action,
            probs.log_prob(action).sum(dim=(1, 2)),
            probs.entropy().sum(dim=(1, 2)),
            self.critic(nobs),
        )


class ResidualMLPAgent(MLPStateActor):
    def __init__(self, device, normalizer, cfg):
        super().__init__(
            device,
            normalizer,
            cfg,
        )
        self.config = cfg

        self.value_head = nn.Sequential(
            layer_init(nn.Linear(self.model.layers[-1].in_features, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=0.1),
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, 1, self.action_dim) * -4.5)

    def training_obs(self, batch: dict, flatten: bool = True):
        # The robot state is already normalized in the dataset
        robot_state = batch["robot_state"]
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        # The parts poses are already normalized in the dataset
        parts_poses = batch["parts_poses"]
        nparts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, nparts_poses], dim=-1)

        if flatten:
            # (n_envs, obs_horizon, obs_dim) --> (n_envs, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    def get_value(self, nobs: torch.Tensor):
        representation = self.model.forward_base(nobs)
        return self.value_head(representation)

    def get_action_and_value(self, nobs: torch.Tensor, action=None):
        # bp()
        representation = self.model.forward_base(nobs)

        values = self.value_head(representation)
        action_mean = self.model.action_head(representation).reshape(
            nobs.shape[0], self.pred_horizon, self.action_dim
        )

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            naction = probs.sample()
            # naction = action_mean
        else:
            naction = self.normalizer(action, "action", forward=True)

        action = self.normalizer(naction, "action", forward=False)

        return (
            action,
            # Sum over all the dimensions after the batch dimension
            probs.log_prob(naction).sum(dim=(1, 2)),
            probs.entropy().sum(dim=(1, 2)),
            values,
        )


class SmallAgentSimple(nn.Module):
    action_horizon: int = 1

    def __init__(self, obs_shape, action_shape):
        super().__init__()
        # bp()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=1),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(action_shape)) * 0)

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        # bp()
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(obs),
        )
