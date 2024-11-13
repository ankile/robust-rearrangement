from collections import deque
import hydra
from omegaconf import DictConfig
from src.behavior.diffusion import DiffusionPolicy
from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.models.residual import ResidualPolicy
import torch
import torch.nn as nn


from ipdb import set_trace as bp  # noqa
from typing import Dict, Union

from collections import namedtuple

ResidualTrainingValues = namedtuple(
    "ResidualTrainingValues",
    [
        "residual_naction_samp",
        "residual_naction_mean",
        "logprob",
        "entropy",
        "value",
        "env_action",
        "next_residual_nobs",
    ],
)


class ResidualDiffusionPolicy(DiffusionPolicy):

    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ) -> None:
        assert cfg.observation_type == "state"

        super().__init__(device, cfg)

        # TODO: Reconsider the way we deal with this
        # E.g., can we separate out this so that it's not in the base class to be overwritten like this?
        # Also, is there a way that's (a) more efficient and (b) allows us to reset just a subset of environments?
        self.actions = None
        self.observations = deque(maxlen=self.obs_horizon)
        self.base_nactions = deque(maxlen=self.action_horizon)

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

    def load_base_state_dict(self, path: str):
        base_state_dict = torch.load(path)
        if "model_state_dict" in base_state_dict:
            base_state_dict = base_state_dict["model_state_dict"]

        # Load the model weights
        base_model_state_dict = {
            key[len("model.") :]: value
            for key, value in base_state_dict.items()
            if key.startswith("model.")
        }
        self.model.load_state_dict(base_model_state_dict)

        # Load normalizer parameters
        base_normalizer_state_dict = {
            key[len("normalizer.") :]: value
            for key, value in base_state_dict.items()
            if key.startswith("normalizer.")
        }
        self.normalizer.load_state_dict(base_normalizer_state_dict)

    def compute_loss(
        self, batch: Dict[str, torch.Tensor], base_only: bool = True
    ) -> torch.Tensor:
        # Cut off the unused observations before passing to the bc loss
        bc_loss, losses = super().compute_loss(batch)
        if base_only:
            return bc_loss, losses

        # Predict the action
        with torch.no_grad():
            nobs = self._training_obs(batch, flatten=self.flatten_obs)
            naction = self._normalized_action(nobs)

        residual_nobs = torch.cat([batch["obs"], naction], dim=-1)
        gt_residual_naction = batch["action"] - naction

        # Don't start supervising the residual until the base model has started converging
        if bc_loss > 0.03:
            return bc_loss, losses

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

        # Concatenate the state and base action
        nobs = nobs.flatten(start_dim=1)
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

    @torch.no_grad()
    def base_action_normalized(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        action = super().action(obs)
        return self.normalizer(action, "action", forward=True)

    def process_obs(self, obs: Dict[str, torch.Tensor]):
        # Robot state is [pos, ori_quat, pos_vel, ori_vel, gripper]
        robot_state = obs["robot_state"]

        # Parts poses is [pos, ori_quat] for each part
        parts_poses = obs["parts_poses"]

        # Make the robot state have 6D proprioception
        if robot_state.shape[-1] == 14:
            robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        robot_state = self.normalizer(robot_state, "robot_state", forward=True)
        parts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        obs = torch.cat([robot_state, parts_poses], dim=-1)

        # Clamp the observation to be bounded to [-5, 5]
        obs = torch.clamp(obs, -3, 3)

        return obs

    def get_action_and_value(
        self,
        obs: Union[Dict[str, torch.Tensor], torch.Tensor],
        action: torch.Tensor = None,
        eval: bool = False,
    ) -> ResidualTrainingValues:
        raise NotImplementedError
        if isinstance(obs, dict):
            # Get the base normalized action
            base_naction = self.base_action_normalized(obs)

            # Process the obs for the residual policy
            next_nobs = self.process_obs(obs)
            next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)

        else:
            assert obs.shape[-1] == self.residual_policy.obs_dim
            next_residual_nobs = obs

        # Get the residual action
        residual_naction_samp, logprob, ent, value, residual_naction_mean = (
            self.residual_policy.get_action_and_value(next_residual_nobs, action=action)
        )

        residual_naction = residual_naction_mean if eval else residual_naction_samp

        if action is None:
            env_naction = (
                base_naction + residual_naction * self.residual_policy.action_scale
            )
            env_action = self.normalizer(env_naction, "action", forward=False)
        else:
            env_action = action

        return ResidualTrainingValues(
            residual_naction_samp=residual_naction_samp,
            residual_naction_mean=residual_naction_mean,
            logprob=logprob,
            entropy=ent,
            value=value,
            env_action=env_action,
            next_residual_nobs=next_residual_nobs,
        )

    def get_value(self, residual_nobs) -> torch.Tensor:
        return self.residual_policy.get_value(residual_nobs)

    def action_normalized(self, obs: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def correct_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict the correction to the action given the observation and the action
        """
        nobs = self.process_obs(obs)
        naction = self.normalizer(action, "action", forward=True)

        residual_nobs = torch.cat([nobs, naction], dim=-1)

        naction_corrected = (
            naction
            + self.residual_policy.actor_mean(residual_nobs)
            * self.residual_policy.action_scale
        )

        return self.normalizer(naction_corrected, "action", forward=False)

    def reset(self):
        """
        Reset the actor
        """
        self.base_nactions.clear()
        self.observations.clear()
        if self.actions is not None:
            self.actions.clear()

    @property
    def actor_parameters(self):
        return [
            p for n, p in self.residual_policy.named_parameters() if "critic" not in n
        ]

    @property
    def critic_parameters(self):
        return [p for n, p in self.residual_policy.named_parameters() if "critic" in n]

    @property
    def base_actor_parameters(self):
        """
        Return the parameters of the base model (actor only)
        """
        return [
            p
            for n, p in self.model.named_parameters()
            if not (n.startswith("encoder1.") or n.startswith("encoder2."))
        ]
