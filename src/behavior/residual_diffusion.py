from collections import deque
import hydra
from omegaconf import DictConfig
from src.behavior.diffusion import DiffusionPolicy
from src.models.residual import ResidualPolicy
import torch
import torch.nn as nn


from ipdb import set_trace as bp  # noqa
from typing import Dict, Union


class ResidualDiffusionPolicy(DiffusionPolicy):

    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ) -> None:
        super().__init__(device, cfg)

        # TODO: Reconsider the way we deal with this
        # E.g., can we separate out this so that it's not in the base class to be overwritten like this?
        # Also, is there a way that's (a) more efficient and (b) allows us to reset just a subset of environments?
        self.observations = deque(maxlen=self.obs_horizon)
        self.base_nactions = deque(maxlen=self.action_horizon)

        if (wts := cfg.actor.get("base_bc_wts", None)) is not None:
            print(f"Loading base bc weights from {wts}")
            self.load_state_dict(torch.load(wts))

            # Freeze the base policy
            for param in self.parameters():
                param.requires_grad = False

        # Make the residual layers:
        # This is an MLP that takes in the state and predicted action
        # and outputs the residual to be added to the predicted action
        # NOTE: What about having a ensemble of residual layers?
        # They're cheap to compute and we can use them to both improve the
        # performance of the policy and to estimate the uncertainty of the
        # policy.
        self.residual_policy: ResidualPolicy = hydra.utils.instantiate(
            cfg.actor.residual_policy,
            obs_shape=(self.timestep_obs_dim,),
            action_shape=(self.action_dim,),
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Cut off the unused observations before passing to the bc loss
        bc_loss, losses = super().compute_loss(batch)

        # Predict the action
        with torch.no_grad():
            nobs = self._training_obs(batch, flatten=self.flatten_obs)
            naction = self._normalized_action(nobs)

        residual_nobs = torch.cat([batch["obs"], naction], dim=-1)
        gt_residual_naction = batch["action"] - naction

        # Residual loss
        residual_loss = self.residual_policy.bc_loss(residual_nobs, gt_residual_naction)

        # Make a dictionary of losses for logging
        losses["residual_loss"] = residual_loss.item()

        return bc_loss + residual_loss, losses

    @torch.no_grad()
    def action(self, obs: Dict[str, torch.Tensor]):
        """
        Predict the action given the batch of observations
        """
        self.observations.append(obs)

        # Normalize observations
        nobs = self._normalized_obs(self.observations, flatten=self.flatten_obs)

        if not self.base_nactions:
            # If there are no base actions, predict the action
            base_nactioon_pred = self._normalized_action(nobs)

            # Add self.action_horizon base actions
            start = self.obs_horizon - 1 if self.predict_past_actions else 0
            end = start + self.action_horizon
            for i in range(start, end):
                self.base_nactions.append(base_nactioon_pred[:, i, :])

        # Pop off the next base action
        base_naction = self.base_nactions.popleft()

        # return self.normalizer(base_naction, "action", forward=False)

        # Concatenate the state and base action
        residual_nobs = torch.cat([nobs, base_naction], dim=-1)

        # Predict the residual (already scaled)
        residual = self.residual_policy.get_action(residual_nobs)

        # Add the residual to the base action
        naction = base_naction + residual

        # Denormalize and return the action
        return self.normalizer(naction, "action", forward=False)

    @torch.no_grad()
    def action_pred(self, batch):
        nobs = self._training_obs(batch, flatten=self.flatten_obs)
        naction = self._normalized_action(nobs)

        residual_nobs = torch.cat([batch["obs"], naction], dim=-1)
        residual = self.residual_policy.get_action(residual_nobs)

        return self.normalizer(naction + residual, "action", forward=False)
