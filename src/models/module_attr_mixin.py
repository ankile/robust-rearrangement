import torch.nn as nn


class ModuleAttrMixin(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
