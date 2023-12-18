import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple MLP with a single hidden layer."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout

        self.layers = nn.ModuleList()
        in_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            in_dim = dim

        self.layers.append(nn.Linear(in_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
