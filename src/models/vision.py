import torch
import transformers
from vip import load_vip
from r3m import load_r3m
from ipdb import set_trace as bp


def get_encoder(encoder_name, freeze=True, device="cuda"):
    if encoder_name == "dinov2-base":
        return DinoEncoder(size="base", freeze=freeze, device=device)
    if encoder_name == "dinov2-large":
        return DinoEncoder(size="large", freeze=freeze, device=device)
    if encoder_name == "dinov2-small":
        return DinoEncoder(size="small", freeze=freeze, device=device)
    elif encoder_name == "r3m-50":
        return R3MEncoder(size=50, freeze=freeze, device=device)
    else:
        raise ValueError(f"Unknown encoder name: {encoder_name}")


class DinoEncoder(torch.nn.Module):
    def __init__(self, size="base", freeze=True, device="cuda"):
        super().__init__()
        assert size in ["small", "base", "large", "giant"]
        self.device = device

        model_name = f"facebook/dinov2-{size}"
        self.trans = transformers.AutoImageProcessor.from_pretrained(model_name)
        self.model = transformers.AutoModel.from_pretrained(model_name).to(self.device)
        self.encoding_dim = self.model.config.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

    def forward(self, x):
        # Move channels to the front
        x = x.permute(0, 3, 1, 2)
        x = self.trans(x, return_tensors="pt").pixel_values.to(self.device)
        x = self.model(x).pooler_output
        return x


class VIPEncoder(torch.nn.Module):
    def __init__(self, freeze=True, device="cuda", *args, **kwargs) -> None:
        super().__init__()
        self.device = device
        self.model = load_vip(device=device).module
        self.encoding_dim = self.model.convnet.fc.out_features

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

    # Expect input to be a batch of images of shape (batch_size, 224, 224, 3) in range [0, 255]
    def forward(self, x):
        # Move channels to the front
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        return x


class R3MEncoder(torch.nn.Module):
    def __init__(self, size=50, freeze=True, device="cuda", *args, **kwargs) -> None:
        super().__init__()
        assert size in [18, 34, 50]

        self.device = device
        self.model = load_r3m(modelid=f"resnet{size}", device=device).module.to(device)
        self.encoding_dim = self.model.convnet.layer4[2].conv3.out_channels

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

    # Expect input to be a batch of images of shape (batch_size, 224, 224, 3) in range [0, 255]
    def forward(self, x):
        # Move channels to the front
        x = x.permute(0, 3, 1, 2)
        x = self.model(x)
        return x
