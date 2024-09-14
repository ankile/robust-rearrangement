import torch
import torch.nn as nn
import torchvision

# torchvision.disable_beta_transforms_warning()

from torchvision import transforms

# from torchvision.transforms import v2 as transforms
from ipdb import set_trace as bp  # noqa


class FrontCameraTransform(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode

        margin = 20
        crop_size = (224, 224)
        input_size = (240, 320)

        self.transform_train = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                ),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
                transforms.CenterCrop((input_size[0], input_size[1] - 2 * margin)),
                transforms.RandomCrop(crop_size),
            ]
        )
        self.transform_eval = transforms.CenterCrop(crop_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[2:] == (240, 320), f"Invalid input shape: {x.shape}"
        if self.mode == "train":
            return self.transform_train(x)
        elif self.mode == "eval":
            return self.transform_eval(x)

        raise ValueError(f"Invalid mode: {self.mode}")

    def train(self, mode=True):
        super().train(mode)
        self.mode = "train" if mode else "eval"

    def eval(self):
        super().eval()
        self.mode = "eval"


class WristCameraTransform(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode

        self.transform_train = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                ),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0)),
                transforms.Resize((224, 224), antialias=True),
            ]
        )
        self.transform_eval = transforms.Resize((224, 224), antialias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "train":
            return self.transform_train(x)
        elif self.mode == "eval":
            return self.transform_eval(x)

        raise ValueError(f"Invalid mode: {self.mode}")

    def train(self, mode=True):
        super().train(mode)
        self.mode = "train" if mode else "eval"

    def eval(self):
        super().eval()
        self.mode = "eval"
