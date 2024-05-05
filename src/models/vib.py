import torch
import torch.nn as nn


class VIB(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc_mu = nn.Linear(z_dim, hidden_dim)
        self.fc_log_var = nn.Linear(z_dim, hidden_dim)

    def train_sample(self, z):
        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z_compressed = mu + eps * std
        return z_compressed, mu, log_var

    def test_sample(self, z):
        mu = self.fc_mu(z)
        return mu

    def forward(self, z):
        return self.test_sample(z)

    @staticmethod
    def kl_divergence(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
