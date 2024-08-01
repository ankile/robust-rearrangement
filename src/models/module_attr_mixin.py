import torch.nn as nn


class ModuleAttrMixin(nn.Module):

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
