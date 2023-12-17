import torch
import torch.nn as nn
from typing import Union
from collections import deque
from ipdb import set_trace as bp  # noqa

from src.behavior.base import Actor
from src.models.mlp import MLP
from src.models.vision import get_encoder
from src.data.normalizer import StateActionNormalizer


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
            output_dim=self.action_dim * self.action_horizon,
            hidden_dims=config.actor_hidden_dims,
            dropout=config.actor_dropout,
        ).to(device)

        self.dropout = (
            nn.Dropout(config.feature_dropout) if config.feature_dropout else None
        )

    def __post_init__(self, *args, **kwargs):
        self.print_model_params()

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:.2e}")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params:.2e} parameters")

    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations
        nobs = self._normalized_obs(obs)

        # Predict normalized action
        naction = self.model(nobs).reshape(
            nobs.shape[0], self.action_horizon, self.action_dim
        )

        # unnormalize action
        # (B, pred_horizon, action_dim)
        action_pred = self.normalizer(naction, "action", forward=False)

        # For now, only return the first action
        return action_pred[:, 0, :]

    # === Training ===
    def compute_loss(self, batch):
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch)

        # Apply Dropout to the observation conditioning if specified
        if self.dropout:
            obs_cond = self.dropout(obs_cond)

        # Action already normalized in the dataset
        # naction = normalize_data(batch["action"], stats=self.stats["action"])
        naction = batch["action"][:, : self.action_horizon, :]

        # forward pass
        naction_pred = self.model(obs_cond).reshape(
            naction.shape[0], self.action_horizon, self.action_dim
        )

        loss = nn.functional.mse_loss(naction_pred, naction)

        return loss
