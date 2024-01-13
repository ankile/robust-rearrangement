import torch
import torch.nn as nn
from src.models.mlp import MLP
from src.behavior.base import PostInitCaller

from ipdb import set_trace as bp  # noqa


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


class CriticModule(nn.Module, metaclass=PostInitCaller):
    """A model to encapsulate both Q and V functions"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        action_horizon: int,
        expectile: float,
        discount: float,
        critic_hidden_dims: list,
        critic_dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        # Add hyperparameters specific to learning value functions implicitly
        self.expectile = expectile
        self.discount = discount

        # Add networks for the Q function
        self.q_network = DoubleCritic(
            state_dim=obs_dim * obs_horizon,
            action_dim=action_dim * self.action_horizon,
            hidden_dims=critic_hidden_dims,
            dropout=critic_dropout,
        ).to(device)

        self.q_target_network = DoubleCritic(
            state_dim=obs_dim * obs_horizon,
            action_dim=action_dim * self.action_horizon,
            hidden_dims=critic_hidden_dims,
            dropout=critic_dropout,
        ).to(device)

        # Turn off gradients for the target network
        for param in self.q_target_network.parameters():
            param.requires_grad = False

        # Add networks for the value function
        self.value_network = ValueNetwork(
            input_dim=obs_dim * obs_horizon,
            hidden_dims=critic_hidden_dims,
            dropout=critic_dropout,
        ).to(device)

    def __post_init__(self, *args, **kwargs):
        self.print_model_params()

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params / 1_000_000:.2f}M")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params / 1_000_000:.2f}M parameters")

    def _flat_action(self, action):
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        naction = action[:, start:end, :].flatten(start_dim=1)
        return naction

    def _training_obs(self, batch):
        """
        This function expects the observations to be already-embedded feature observations,
        not raw images for now
        """
        # The robot state is already normalized in the dataset
        nrobot_state = batch["robot_state"]

        # All observations already normalized in the dataset
        feature1 = batch["feature1"]
        feature2 = batch["feature2"]

        # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
        nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
        nobs = nobs.flatten(start_dim=1)

        return nobs

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
        terminal = batch["terminal"]

        with torch.no_grad():
            next_v = self.value_network(next_obs).squeeze(-1)

        target_q = batch["reward"] + self.discount * next_v * (1 - terminal)

        q1, q2 = self.q_network(curr_obs, naction)

        q1_loss = nn.functional.mse_loss(q1.squeeze(-1), target_q)
        q2_loss = nn.functional.mse_loss(q2.squeeze(-1), target_q)

        return (q1_loss + q2_loss) / 2

    def compute_loss(self, batch):
        q_loss = self._q_loss(batch)
        value_loss = self._value_loss(batch)

        return q_loss, value_loss

    @torch.no_grad()
    def polyak_update_target(self, tau):
        for param, target_param in zip(
            self.q_network.parameters(), self.q_target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def q_value(self, nobs, naction):
        return torch.min(*self.q_network(nobs, naction)).squeeze(-1)

    def value(self, nobs):
        return self.value_network(nobs).squeeze(-1)

    def action_weights(self, nobs, nactions):
        """
        Compute the action weights for the action selection
        It takes in a batch of observations and a batch of actions
        A batch is equal to the number of environments used in rollouts
        Each element in a batch of actions is a set of actions for a single environment
        """
        n_action_samples = nactions.shape[0]

        # bp()

        # 3. Compute w^\tau_2(s, a_i) = Q(s, a_i) - V(s)
        qs = torch.min(
            *self.q_network(
                nobs.unsqueeze(0).expand(n_action_samples, -1, -1),
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

        return tau_weights
