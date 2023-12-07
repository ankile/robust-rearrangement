import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """A network that predicts the value of a state."""

    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, output_dim=1, dropout=dropout)

    def forward(self, x):
        return self.mlp(x)


class Critic(nn.Module):
    """A critic that predicts the value of a state-action pair."""

    def __init__(self, state_dim, action_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.mlp = MLP(
            state_dim + action_dim, hidden_dims, output_dim=1, dropout=dropout
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.mlp(x)


class DoubleCritic(nn.Module):
    """A critic that predicts the value of a state-action pair."""

    def __init__(self, state_dim, action_dim, hidden_dims, dropout=0.0):
        super().__init__()
        self.critic1 = Critic(state_dim, action_dim, hidden_dims, dropout=dropout)
        self.critic2 = Critic(state_dim, action_dim, hidden_dims, dropout=dropout)

    def forward(self, state, action):
        return self.critic1(state, action), self.critic2(state, action)
