from typing import Tuple

import numpy as np

from src.models.utils import PrintParamCountMixin
import torch
import torch.nn as nn
from torch.distributions import Normal

from ipdb import set_trace as bp


def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )  # Use relu for Swish
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)

    return layer


def build_mlp(
    input_dim,
    hidden_sizes,
    output_dim,
    activation,
    output_std=1.0,
    bias_on_last_layer=True,
):
    act_func = getattr(nn, activation)
    layers = []
    layers.append(
        layer_init(nn.Linear(input_dim, hidden_sizes[0]), nonlinearity=activation)
    )
    layers.append(act_func())
    for i in range(1, len(hidden_sizes)):
        layers.append(
            layer_init(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nonlinearity=activation
            )
        )
        layers.append(act_func())
    layers.append(
        layer_init(
            nn.Linear(hidden_sizes[-1], output_dim, bias=bias_on_last_layer),
            std=output_std,
            nonlinearity="Tanh",
        )
    )
    return nn.Sequential(*layers)


class ResidualPolicy(nn.Module, PrintParamCountMixin):
    def __init__(
        self,
        obs_shape,
        action_shape,
        actor_hidden_size=512,
        actor_num_layers=2,
        critic_hidden_size=512,
        critic_num_layers=2,
        actor_activation="SiLU",
        critic_activation="SiLU",
        init_logstd=-3,
        action_head_std=0.01,
        action_scale=0.1,
    ):
        """
        Args:
            obs_shape: the shape of the observation (i.e., state + base action)
            action_shape: the shape of the action (i.e., residual, same size as base action)
            actor_hidden_sizes: list of hidden layer sizes for the actor network
            critic_hidden_sizes: list of hidden layer sizes for the critic network
            activation: activation function to use (e.g., nn.ReLU, nn.Tanh)
        """
        super().__init__()

        self.action_dim = action_shape[-1]
        self.obs_dim = np.prod(obs_shape) + np.prod(action_shape)
        self.action_scale = action_scale

        self.actor_mean = build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=[actor_hidden_size] * actor_num_layers,
            output_dim=np.prod(action_shape),
            activation=actor_activation,
            output_std=action_head_std,
            bias_on_last_layer=False,
        )

        self.critic = build_mlp(
            input_dim=self.obs_dim,
            hidden_sizes=[critic_hidden_size] * critic_num_layers,
            output_dim=1,
            activation=critic_activation,
            output_std=0.01,
            bias_on_last_layer=True,
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * init_logstd)

        self.print_model_params()

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(
        self,
        nobs: torch.Tensor,
        action: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean: torch.Tensor = self.actor_mean(nobs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(dim=1),
            probs.entropy().sum(dim=1),
            self.critic(nobs),
            action_mean,
        )


class BiggerResidualPolicy(ResidualPolicy):

    def __init__(self, obs_shape, action_shape, init_logstd=0, action_head_std=0.01):
        """
        Args:
            obs_shape: the shape of the observation (i.e., state + base action)
            action_shape: the shape of the action (i.e., residual, same size as base action)
        """
        super().__init__(obs_shape, action_shape, init_logstd)

        self.action_dim = action_shape[-1]
        self.obs_dim = np.prod(obs_shape) + np.prod(action_shape)

        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(self.obs_dim, 1024), nonlinearity="swish"),
            nn.LayerNorm(1024),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024), nonlinearity="swish"),
            nn.LayerNorm(1024),
            nn.SiLU(),
            layer_init(
                nn.Linear(1024, np.prod(action_shape), bias=False), std=action_head_std
            ),
        )

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(self.obs_dim, 1024), nonlinearity="swish"),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            layer_init(nn.Linear(1024, 1024), nonlinearity="swish"),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            layer_init(nn.Linear(1024, 1), std=1.0),
        )
