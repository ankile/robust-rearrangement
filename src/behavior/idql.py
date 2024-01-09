from collections import deque
import torch
import torch.nn as nn
from src.data.normalizer import StateActionNormalizer
from src.models.vision import get_encoder
from src.models.unet import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.models.value import CriticModule
from src.behavior.base import Actor
from src.behavior.diffusion_policy import DiffusionPolicy

from ipdb import set_trace as bp  # noqa
from typing import Union


class ImplicitQActor(DiffusionPolicy):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: StateActionNormalizer,
        config,
    ) -> None:
        super().__init__(device, encoder_name, freeze_encoder, normalizer, config)

        # Add hyperparameters specific to IDQL
        self.n_action_samples = config.n_action_samples

        self.critic_module = CriticModule(
            discount=config.discount,
            expectile=config.expectile,
            critic_hidden_dims=config.critic_hidden_dims,
            critic_dropout=config.critic_dropout,
            device=device,
        )

    # === Inference ===
    def _sample_action_pred(self, nobs):
        # 2. Sample action actions a_i ~ pi(a_i | s_i) for i = 1, ..., N
        # But do it in parallel by packing all environments * action samples into a single batch
        # The observation will be properly handled in the call to self._normalized_action
        nstacked_obs = nobs.unsqueeze(0).expand(self.n_action_samples, -1, -1)
        nactions = self._normalized_action(
            nstacked_obs.reshape(self.n_action_samples * nobs.shape[0], -1)
        ).reshape(self.n_action_samples, nobs.shape[0], self.pred_horizon, -1)
        nactions = nactions[:, :, : self.action_horizon, :]

        # 3. Compute w^\tau_2(s, a_i) = Q(s, a_i) - V(s)
        tau_weights = self.critic_module(nobs, nactions)

        # 4. Sample the action out of the candidates
        probabilities = torch.softmax(tau_weights, dim=0)
        sample_idx = torch.multinomial(probabilities.T, num_samples=1)
        env_indices = torch.arange(
            nactions.size(1), device=sample_idx.device
        ).unsqueeze(1)
        naction = nactions[sample_idx, env_indices, :, :].squeeze(1)

        # 5. Unnormalize action and play the action
        # (B, pred_horizon, action_dim)
        action_pred = self.normalizer(naction, "action", forward=False)

        # Add the actions to the queue
        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        actions = deque()
        for i in range(start, end):
            actions.append(action_pred[:, i, :])

        return actions
