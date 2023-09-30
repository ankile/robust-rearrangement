# Code borrowed from https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
# Credit to https://github.com/jvanvugt for implmenting ReverseGradient from:

# @article{ganin2014unsupervised,
#   title={Unsupervised domain adaptation by backpropagation. arXiv},
#   author={Ganin, Yaroslav and Lempitsky, V},
#   journal={arXiv preprint arXiv:1409.7495},
#   year={2014}
# }


import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DomainClassifier, self).__init__()

        self.net = nn.Sequential(
            GradientReversal(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)
