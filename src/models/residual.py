from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, nonlinearity="relu", std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        if nonlinearity == "relu":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "swish":
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )  # Use relu for Swish
        elif nonlinearity == "tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)

    return layer


class ResidualPolicy(nn.Module):
    def __init__(self, obs_shape, action_shape, init_logstd=0, action_head_std=0.01):
        """
        Args:
            obs_shape: the shape of the observation (i.e., state + base action)
            action_shape: the shape of the action (i.e., residual, same size as base action)
        """
        super().__init__()

        self.action_dim = action_shape[-1]
        self.obs_dim = np.prod(obs_shape) + np.prod(action_shape)

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 512), nonlinearity="swish"),
            nn.SiLU(),
            layer_init(nn.Linear(512, 512), nonlinearity="swish"),
            nn.SiLU(),
            layer_init(
                nn.Linear(512, np.prod(action_shape), bias=False),
                std=action_head_std,
                nonlinearity="tanh",
            ),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 512), nonlinearity="tanh"),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512), nonlinearity="tanh"),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0, nonlinearity="tanh"),
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * init_logstd)

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
