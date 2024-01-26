import torch
import torch.nn as nn
from torchvision import transforms


# Set image transforms
margin = 20
crop_size = (224, 224)
input_size = (240, 320)


class FrontCameraTransform(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode

        margin = 20
        crop_size = (224, 224)
        input_size = (240, 320)

        self.transform_train = transforms.Compose(
            [
                transforms.CenterCrop((input_size[0], input_size[1] - 2 * margin)),
                transforms.RandomCrop(crop_size),
            ]
        )
        self.transform_eval = transforms.CenterCrop(crop_size)

    def forward(self, x):
        if self.mode == "train":
            return self.transform_train(x)
        elif self.mode == "eval":
            return self.transform_eval(x)

        raise ValueError(f"Invalid mode: {self.mode}")

    def train(self, mode=True):
        self.mode = "train"
        super().train(mode)

    def eval(self):
        self.mode = "eval"
        super().eval()


class WristCameraTransform(nn.Module):
    def __init__(self, mode="train"):
        super().__init__()
        self.mode = mode

        self.transform_train = transforms.Resize((224, 224), antialias=True)
        self.transform_eval = transforms.Resize((224, 224), antialias=True)

    def forward(self, x):
        if self.mode == "train":
            return self.transform_train(x)
        elif self.mode == "eval":
            return self.transform_eval(x)

        raise ValueError(f"Invalid mode: {self.mode}")

    def train(self, mode=True):
        self.mode = "train"
        super().train(mode)

    def eval(self):
        self.mode = "eval"
        super().eval()
