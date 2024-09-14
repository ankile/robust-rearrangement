from omegaconf import DictConfig
from src.models.residual import build_mlp
import torch
import torch.nn as nn
from typing import Tuple, Union
from ipdb import set_trace as bp  # noqa

from src.behavior.base import Actor
from src.models.mlp import MLP


import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class MLPActor(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ) -> None:
        super().__init__(device, cfg)
        actor_cfg = cfg.actor

        # MLP specific parameters
        self.model = MLP(
            input_dim=self.timestep_obs_dim * self.obs_horizon,
            output_dim=self.action_dim * self.pred_horizon,
            hidden_dims=actor_cfg.hidden_dims,
            dropout=actor_cfg.dropout,
            residual=actor_cfg.residual,
        ).to(device)

        if "critic" in actor_cfg:
            self.critic = build_mlp(
                input_dim=self.obs_dim,
                hidden_sizes=[actor_cfg.critic.hidden_size]
                * actor_cfg.critic.num_layers,
                output_dim=1,
                activation=actor_cfg.critic.activation,
                output_std=actor_cfg.critic.last_layer_std,
                bias_on_last_layer=True,
                last_layer_bias_const=actor_cfg.critic.last_layer_bias_const,
            )

            if actor_cfg.critic.last_layer_activation is not None:
                self.critic.add_module(
                    "output_activation",
                    getattr(nn, actor_cfg.critic.last_layer_activation)(),
                )

                print(self.critic)

            self.actor_mean = nn.Sequential(
                self.model,
                nn.Unflatten(1, (self.pred_horizon, self.action_dim)),
            )

            self.actor_logstd = nn.Parameter(
                torch.ones(1, 1, self.action_dim) * actor_cfg.init_logstd
            )

    # === Inference ===
    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        naction = self.model(nobs).reshape(
            nobs.shape[0], self.pred_horizon, self.action_dim
        )

        return naction

    # === Training ===
    def compute_loss(self, batch):
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=True)

        # Action already normalized in the dataset
        # These actions are the exact ones we should predict, i.e., the
        # handling of predicting past actions or not is also handled in the dataset class
        naction = batch["action"]

        # forward pass
        naction_pred = self.model(obs_cond).reshape(
            naction.shape[0], self.pred_horizon, self.action_dim
        )

        # The loss function does not have reduction on by default, need to take the mean
        loss = self.loss_fn(naction_pred, naction).mean()

        losses = {"bc_loss": loss.item()}

        return loss, losses

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(self, nobs: torch.Tensor, action=None):
        action_mean: torch.Tensor = self.actor_mean(nobs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.rsample()

        return (
            action,
            probs.log_prob(action).sum(dim=(1, 2)),
            probs.entropy().sum(dim=(1, 2)),
            self.critic(nobs),
        )

    def load_bc_weights(self, bc_weights_path):
        wts = torch.load(bc_weights_path)["model_state_dict"]

        # Filter out keys not starting with "model"
        model_wts = {k: v for k, v in wts.items() if k.startswith("model")}

        # Change the "model"
        model_wts = {k.replace("model.", ""): v for k, v in model_wts.items()}

        print(self.model.load_state_dict(model_wts, strict=True))

        # Extract the normalizer state dict from the overall state dict
        normalizer_state_dict = {
            key[len("normalizer.") :]: value
            for key, value in wts.items()
            if key.startswith("normalizer.")
        }

        # Load the normalizer state dict
        print(self.normalizer.load_state_dict(normalizer_state_dict))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SmallMLPAgent(nn.Module):
    def __init__(self, obs_shape: tuple, action_shape: tuple, init_logstd=0):
        super().__init__()

        assert (
            len(action_shape) == 2
        ), "Actions must be of shape (action_horizon, action_dim)"

        self.action_horizon, self.action_dim = action_shape

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
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
            nn.Unflatten(1, action_shape),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, 1, self.action_dim) * init_logstd
        )

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(self, nobs: torch.Tensor, action=None):
        action_mean: torch.Tensor = self.actor_mean(nobs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.rsample()

        return (
            action,
            probs.log_prob(action).sum(dim=(1, 2)),
            probs.entropy().sum(dim=(1, 2)),
            self.critic(nobs),
        )


class BigMLPAgent(SmallMLPAgent):
    """
    A bigger agent with more hidden layers than the SmallMLPAgent
    """

    def __init__(self, obs_shape, action_shape, init_logstd=0):
        super().__init__(obs_shape, action_shape, init_logstd)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(action_shape)), std=0.01),
            nn.Unflatten(1, action_shape),
        )


class ResidualMLPAgent(nn.Module):

    def __init__(
        self, obs_shape: tuple, action_shape: tuple, init_logstd=0, dropout=0.1
    ):
        super().__init__()

        assert (
            len(action_shape) == 2
        ), "Actions must be of shape (action_horizon, action_dim)"

        self.action_horizon, self.action_dim = action_shape

        self.backbone_emb_dim = 1024

        self.backbone = MLP(
            input_dim=np.array(obs_shape).prod(),
            output_dim=self.backbone_emb_dim,
            hidden_dims=[1024] * 5,
            dropout=0.0,
            residual=True,
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.backbone_emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.value_head.apply(self.init_weights)

        self.action_head = nn.Sequential(
            nn.Linear(self.backbone_emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, np.prod(action_shape)),
            nn.Unflatten(1, action_shape),
        )

        self.action_head.apply(self.init_weights)

        self.actor_logstd = nn.Parameter(
            torch.ones(1, 1, self.action_dim) * init_logstd
        )

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        representation = self.backbone(nobs)
        return self.value_head(representation)

    def actor_mean(self, nobs: torch.Tensor) -> torch.Tensor:
        representation = self.backbone(nobs)
        return self.action_head(representation)

    def get_action_and_value(
        self, nobs: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        representation: torch.Tensor = self.backbone(nobs)
        action_mean: torch.Tensor = self.action_head(representation)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.rsample()

        return (
            action,
            # The probs are calculated over a whole chunk as a single action
            # (batch, action_horizon, action_dim) -> (batch, )
            probs.log_prob(action).sum(dim=(1, 2)),
            probs.entropy().sum(dim=(1, 2)),
            self.value_head(representation),
        )


class ResidualMLPAgentSeparate(SmallMLPAgent):
    def __init__(self, obs_shape: tuple, action_shape: tuple, init_logstd=0):
        super().__init__(obs_shape, action_shape, init_logstd)

        self.actor_mean = nn.Sequential(
            MLP(
                input_dim=np.array(obs_shape).prod(),
                output_dim=np.prod(action_shape),
                hidden_dims=[1024] * 5,
                dropout=0.0,
                residual=True,
            ),
            nn.Unflatten(1, action_shape),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=0.01),
        )
