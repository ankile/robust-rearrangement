from collections import deque
import torch
import torch.nn as nn
from src.data.normalizer import StateActionNormalizer
from src.models.vision import get_encoder
from src.models.unet import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.models.value import DoubleCritic, ValueNetwork
from src.behavior.base import Actor

from ipdb import set_trace as bp  # noqa
from typing import Union


class ImplicitQActor(DoubleImageActor):
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
        self.expectile = config.expectile
        self.tau = config.q_target_update_step
        self.discount = config.discount
        self.n_action_samples = config.n_action_samples

        # Add networks for the Q function
        self.q_network = DoubleCritic(
            state_dim=self.obs_dim,
            action_dim=self.action_dim * self.action_horizon,
            hidden_dims=config.critic_hidden_dims,
            dropout=config.critic_dropout,
        ).to(device)

        self.q_target_network = DoubleCritic(
            state_dim=self.obs_dim,
            action_dim=self.action_dim * self.action_horizon,
            hidden_dims=config.critic_hidden_dims,
            dropout=config.critic_dropout,
        ).to(device)

        # Turn off gradients for the target network
        for param in self.q_target_network.parameters():
            param.requires_grad = False

        # Add networks for the value function
        self.value_network = ValueNetwork(
            input_dim=self.obs_dim,
            hidden_dims=config.critic_hidden_dims,
            dropout=config.critic_dropout,
        ).to(device)

    def __post_init__(self, *args, **kwargs):
        self.print_model_params()

    def _flat_action(self, action):
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        naction = action[:, start:end, :].flatten(start_dim=1)
        return naction

    def _value_loss(self, batch):
        def loss(diff, expectile=0.8):
            weight = torch.where(
                diff > 0,
                torch.full_like(diff, expectile),
                torch.full_like(diff, 1 - expectile),
            )
            return weight * (diff**2)

        # Compute the value loss
        nobs = self._training_obs(batch["curr_obs"])
        naction = self._flat_action(batch["action"])

        # Compute the Q values
        with torch.no_grad():
            q1, q2 = self.q_target_network(nobs, naction)
            q = torch.min(q1, q2)

        v = self.value_network(nobs)

        # Compute the value loss
        value_loss = loss(q - v, expectile=self.expectile).mean()

        return value_loss

    def _q_loss(self, batch):
        curr_obs = self._training_obs(batch["curr_obs"])
        next_obs = self._training_obs(batch["next_obs"])
        naction = self._flat_action(batch["action"])

        with torch.no_grad():
            next_v = self.value_network(next_obs).squeeze(-1)

        target_q = batch["reward"] + self.discount * next_v

        q1, q2 = self.q_network(curr_obs, naction)

        q1_loss = nn.functional.mse_loss(q1.squeeze(-1), target_q)
        q2_loss = nn.functional.mse_loss(q2.squeeze(-1), target_q)

        return (q1_loss + q2_loss) / 2

    def compute_loss(self, batch):
        bc_loss = super().compute_loss({**batch["curr_obs"], "action": batch["action"]})
        q_loss = self._q_loss(batch)
        value_loss = self._value_loss(batch)

        return bc_loss, q_loss, value_loss

    def polyak_update_target(self, tau):
        with torch.no_grad():
            for param, target_param in zip(
                self.q_network.parameters(), self.q_target_network.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

    # === Inference ===
    @torch.no_grad()
    def action(self, obs: deque):
        # 1. Observe current state s
        nobs = self._normalized_obs(obs)

        # 2. Sample action actions a_i ~ pi(a_i | s_i) for i = 1, ..., N
        # But do it in parallel by packing all environments * action samples into a single batch
        # The observation will be properly handled in the call to self._normalized_action
        nstacked_obs = nobs.unsqueeze(0).expand(self.n_action_samples, -1, -1)
        nactions = self._normalized_action(
            nstacked_obs.reshape(self.n_action_samples * nobs.shape[0], -1)
        ).reshape(self.n_action_samples, nobs.shape[0], self.pred_horizon, -1)
        nactions = nactions[:, :, : self.action_horizon, :]

        # 3. Compute w^\tau_2(s, a_i) = Q(s, a_i) - V(s)
        # Assuming compute_q and compute_v are defined elsewhere in your PyTorch code
        qs = torch.min(
            *self.q_network(
                nobs.unsqueeze(0).expand(self.n_action_samples, -1, -1),
                nactions.flatten(start_dim=2),
            )
        ).squeeze(-1)
        vs = self.value_network(nobs).squeeze(-1)
        adv = qs - vs

        # if self.critic_objective == 'expectile':
        tau_weights = torch.where(
            adv > 0,
            torch.full_like(adv, self.expectile),
            torch.full_like(adv, 1 - self.expectile),
        )
        probabilities = torch.softmax(tau_weights, dim=0)
        sample_idx = torch.multinomial(probabilities.T, num_samples=1)
        env_indices = torch.arange(
            nactions.size(1), device=sample_idx.device
        ).unsqueeze(1)
        naction = nactions[sample_idx, env_indices, :, :].squeeze(1)

        # unnormalize action
        # (B, pred_horizon, action_dim)
        action_pred = self.normalizer(naction, "action", forward=False)

        return action_pred
