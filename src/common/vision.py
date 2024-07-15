import torch
import torch.nn as nn

from torchvision.transforms import transforms

from ipdb import set_trace as bp  # noqa


class FrontCameraTransform(nn.Module):

    def __init__(self, mode="train"):
        super().__init__()
        margin = 20
        crop_size = (224, 224)
        input_size = (240, 320)

        self.transform_train = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                ),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 1.2)),
                transforms.CenterCrop((input_size[0], input_size[1] - 2 * margin)),
                transforms.RandomCrop(crop_size),
                transforms.RandomErasing(value="random", p=0.2),
            ]
        )
        self.transform_eval = transforms.CenterCrop(crop_size)

        self.transform = (
            self.transform_train if mode == "train" else self.transform_eval
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-3:] == (3, 240, 320), x.shape
        return self.transform(x)

    def train(self, mode=True):
        super().train(mode)
        # self.transform = self.transform_train if mode else self.transform_eval

    def eval(self):
        super().eval()
        self.transform = self.transform_eval


class WristCameraTransform(nn.Module):
    transform: nn.Module

    def __init__(self, mode="train"):
        super().__init__()

        self.transform_train = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
                ),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 1.2)),
                transforms.Resize((224, 224), antialias=True),
            ]
        )
        self.transform_eval = transforms.Resize((224, 224), antialias=True)

        self.transform = (
            self.transform_train if mode == "train" else self.transform_eval
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-3:] == (3, 240, 320), x.shape

        return self.transform(x)

    def train(self, mode=True):
        super().train(mode)
        # self.transform = self.transform_train if mode else self.transform_eval

    def eval(self):
        super().eval()
        self.transform = self.transform_eval
