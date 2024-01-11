import torch
import torch.nn as nn
from typing import Union
from collections import deque
from ipdb import set_trace as bp  # noqa

from src.behavior.base import Actor
from src.models.mlp import MLP
from src.models.vision import get_encoder
from src.dataset.normalizer import StateActionNormalizer


class RNNActor(Actor):
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

        # heavily borrowed from
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/robomimic_image_policy.py
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/robomimic_lowdim_policy.py#L25
        # https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/task/tool_hang_image.yaml 

        from robomimic.algo import algo_factory
        from robomimic.algo.algo import PolicyAlgo
        import robomimic.utils.obs_utils as ObsUtils

        # shape_meta = {
        #     'action': {
        #         'shape': self.action_dim
        #     },
        #     'obs': {
        #         'feature': {
        #             'shape': [self.timestep_obs_dim],  # RNN expects inputs in form (B, T, D)
        #             'type': 'low_dim'
        #         }
        #     }
        # }
        # action_dim = shape_meta['action']['shape']
        # obs_shape_meta = shape_meta['obs']  
        # obs_config = {
        #     'low_dim': [],
        #     'rgb': [],
        #     'depth': [],
        #     'scan': []
        # }
        # obs_key_shapes = dict()
        # for key, attr in obs_shape_meta.items():
        #     shape = attr['shape']
        #     obs_key_shapes[key] = list(shape)

        #     type = attr.get('type', 'low_dim')
        #     if type == 'rgb':
        #         obs_config['rgb'].append(key)
        #     elif type == 'low_dim':
        #         obs_config['low_dim'].append(key)
        #     else:
        #         raise RuntimeError(f"Unsupported obs type: {type}")
        # with config.unlocked():
        #     # set config with shape_meta
        #     config.observation.modalities.obs = obs_config

        obs_key = 'obs'
        obs_key_shapes = {obs_key: [self.timestep_obs_dim]}  # RNN expects inputs in form (B, T, D)
        action_dim = self.action_dim

        from src.baseline.robomimic_config_util import get_rm_config
        config = get_rm_config()

        with config.unlocked():
            config.observation.modalities.obs.low_dim = [obs_key]

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # self.model = None
        self.model: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes, #{obs_key: [self.obs_dim]},
            ac_dim=action_dim,
            device=device
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
            obs_dict = {
                "obs": nobs
            }
            naction = self.model.get_action(obs_dict)

            # unnormalize action
            # (B, pred_horizon, action_dim)
            action_pred = self.normalizer(naction, "action", forward=False).reshape(-1, self.action_horizon, self.action_dim)

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
        naction_pred = self.model.nets["policy"](obs_dict=obs_dict)[:, 0].reshape(*naction.shape)

        loss = nn.functional.mse_loss(naction_pred, naction)

        return loss

