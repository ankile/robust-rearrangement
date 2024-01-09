import torch
import torch.nn as nn
from typing import Union
from collections import deque
from ipdb import set_trace as bp  # noqa

from src.behavior.base import Actor
from src.models.mlp import MLP
from src.models.vision import get_encoder
from src.dataset.normalizer import StateActionNormalizer


class MLPActor(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: StateActionNormalizer,
        config,
    ) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        self.action_horizon = config.action_horizon

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        self.obs_horizon = config.obs_horizon
        self.observation_type = config.observation_type
        self.noise_augment = config.noise_augment
        self.freeze_encoder = freeze_encoder
        self.device = device

        # Convert the stats to tensors on the device
        self.normalizer = normalizer.to(device)

        self.encoder1 = get_encoder(encoder_name, freeze=freeze_encoder, device=device)
        self.encoder2 = (
            self.encoder1
            if freeze_encoder
            else get_encoder(encoder_name, freeze=freeze_encoder, device=device)
        )

        self.encoding_dim = self.encoder1.encoding_dim + self.encoder2.encoding_dim
        self.timestep_obs_dim = config.robot_state_dim + self.encoding_dim
        self.obs_dim = self.timestep_obs_dim * self.obs_horizon

        self.model = MLP(
            input_dim=self.obs_dim,
            output_dim=self.action_dim * self.pred_horizon,
            hidden_dims=config.actor_hidden_dims,
            dropout=config.actor_dropout,
        ).to(device)

        self.dropout = (
            nn.Dropout(config.actor_dropout) if config.actor_dropout else None
        )

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
        obs_cond = self._training_obs(batch)

        # Action already normalized in the dataset
        naction = batch["action"]

        # forward pass
        naction_pred = self.model(obs_cond).reshape(
            naction.shape[0], self.pred_horizon, self.action_dim
        )

        loss = nn.functional.mse_loss(naction_pred, naction)

        return loss
