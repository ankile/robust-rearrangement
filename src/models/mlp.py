import torch.nn as nn


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, residual=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout

        self.actvn = nn.ReLU()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for dim in hidden_dims:
            layer = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(dim, dim),
            )
            self.layers.append(layer)

        self.layers.append(nn.Linear(dim, output_dim))
        self.forward = self.forward_res if residual else self.forward_std

    def forward_base(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = self.actvn(x + layer(x))

        return x

    def action_head(self, x):
        return self.layers[-1](x)

    def forward_res(self, x):
        x = self.forward_base(x)
        x = self.action_head(x)
        return x

    def forward_std(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
