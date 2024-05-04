from omegaconf import DictConfig
from src.common.control import RotationMode
import torch
import torch.nn as nn
from typing import Union
from collections import deque
from ipdb import set_trace as bp  # noqa

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils

from src.behavior.base import Actor
from src.models import get_encoder
from src.dataset.normalizer import Normalizer
from src.baseline.robomimic_config_util import get_rm_config


class RNNActor(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        normalizer: Normalizer,
        config: DictConfig,
    ) -> None:
        super().__init__()
        actor_cfg = config.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = (
            10 if config.control.act_rot_repr == RotationMode.rot_6d else 8
        )
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        self.inference_steps = actor_cfg.inference_steps
        self.observation_type = config.observation_type

        # Regularization
        self.feature_noise = config.regularization.get("feature_noise", None)
        self.feature_dropout = config.regularization.get("feature_dropout", None)
        self.feature_layernorm = config.regularization.get("feature_layernorm", None)
        self.state_noise = config.regularization.get("state_noise", False)

        self.device = device

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

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

        # Heavily borrowed from
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/robomimic_image_policy.py
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/robomimic_lowdim_policy.py#L25
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/task/tool_hang_image.yaml

        obs_key = "obs"
        obs_key_shapes = {
            obs_key: [self.timestep_obs_dim]
        }  # RNN expects inputs in form (B, T, D)
        action_dim = self.action_dim

        config = get_rm_config()

        with config.unlocked():
            config.observation.modalities.obs.low_dim = [obs_key]

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # self.model = None
        self.model: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device=device,
        )

    def train_mode(self):
        """
        Set models to train mode
        """
        self.model.set_train()

    def eval_mode(self):
        """
        Set models to eval mode
        """
        self.model.set_eval()

    # === Inference ===
    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations (only want the last one for RNN policy)
        nobs = self._normalized_obs(obs, flatten=False)[:, -1]

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            # Predict normalized action

            # https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/algo/bc.py#L537
            obs_dict = {"obs": nobs}
            naction = self.model.get_action(obs_dict)

            # unnormalize action
            # (B, pred_horizon, action_dim)
            action_pred = self.normalizer(naction, "action", forward=False).reshape(
                -1, self.action_horizon, self.action_dim
            )

            # Add the actions to the queue
            # only take action_horizon number of actions
            start = 0  # first index for RNN policy (I think this is fine?)
            end = start + self.action_horizon
            for i in range(start, end):
                self.actions.append(action_pred[:, i, :])

        # Return the first action in the queue
        return self.actions.popleft()

    # === Training ===
    def compute_loss(self, batch):
        # State already normalized in the dataset (don't flatten for RNN - need time dimension)
        obs_cond = self._training_obs(batch, flatten=False)

        # Action already normalized in the dataset
        naction = batch["action"]

        obs_dict = {"obs": obs_cond}
        naction_pred = self.model.nets["policy"](obs_dict=obs_dict)[:, 0].reshape(
            *naction.shape
        )

        loss = nn.functional.mse_loss(naction_pred, naction)

        return loss
