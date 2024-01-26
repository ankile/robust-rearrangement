import torch
import torch.nn as nn
import torchvision
import transformers


from ipdb import set_trace as bp
from src.models.module_attr_mixin import ModuleAttrMixin
from src.common.pytorch_util import replace_submodules
from src.models.vit import vit_base_patch16


def get_encoder(encoder_name, freeze=True, device="cuda", pretrained=True):
    if encoder_name.startswith("dinov2"):
        return DinoV2Encoder(model_name=encoder_name, freeze=freeze, device=device)
    if encoder_name.startswith("r3m"):
        return R3MEncoder(model_name=encoder_name, freeze=freeze, device=device)
    if encoder_name == "vip":
        return VIPEncoder(freeze=freeze, device=device)
    if encoder_name.startswith("resnet"):
        return ResnetEncoder(
            model_name=encoder_name,
            freeze=freeze,
            device=device,
            use_groupnorm=True,
            pretrained=pretrained,
        )
    if encoder_name == "spatial_softmax":
        return SpatialSoftmaxEncoder(freeze=False, device=device)
    if encoder_name == "dino":
        return DinoEncoder(freeze=freeze, device=device)
    if encoder_name == "mae":
        return MAEEncoder(freeze=freeze, device=device)
    if encoder_name == "voltron":
        return VoltronEncoder(freeze=freeze, device=device)
    raise ValueError(f"Unknown encoder name: {encoder_name}")


# Function borrowed from
# https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/vision/model_getter.py
def get_resnet(model_name, weights=None, **kwargs):
    """
    size: 18, 34, 50
    """
    func = getattr(torchvision.models, model_name)
    resnet = func(weights=weights, **kwargs)
    resnet.encoding_dim = resnet.fc.in_features
    resnet.fc = torch.nn.Identity()
    return resnet


class VisionEncoder(ModuleAttrMixin):
    model: nn.Module

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )


class SpatialSoftmaxEncoder(VisionEncoder):
    def __init__(
        self,
        model_name="ResNet18Conv",
        freeze=True,
        device="cuda",
        use_groupnorm=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        from robomimic.models.obs_core import VisualCore

        self.encoding_dim = 256

        self.model = VisualCore(
            input_shape=[3, 224, 224],
            backbone_class=model_name,
            pool_class="SpatialSoftmax",
            pool_kwargs={"num_kp": 32},
            flatten=True,
            feature_dimension=self.encoding_dim,
        )

        if use_groupnorm:
            self.model = replace_submodules(
                root_module=self.model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        self.model = self.model.to(device)

        if freeze:
            self.freeze()

    # Expect input to be a batch of images of shape (batch_size, 224, 224, 3) in range [0, 255]
    def forward(self, x):
        # Move channels to the front
        x = x.permute(0, 3, 1, 2)
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
        return x


class ResnetEncoder(VisionEncoder):
    def __init__(
        self,
        model_name,
        freeze=True,
        device="cuda",
        use_groupnorm=True,
        pretrained=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert model_name in ["resnet18", "resnet34", "resnet50"]

        weights = "IMAGENET1K_V1" if pretrained else None

        self.model = get_resnet(model_name=model_name, weights=weights)
        self.encoding_dim = self.model.encoding_dim

        if use_groupnorm:
            self.model = replace_submodules(
                root_module=self.model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        self.model = self.model.to(device)

    # Expect input to be a batch of images of shape (batch_size, 224, 224, 3) in range [0, 255]
    def forward(self, x):
        # Move channels to the front
        x = x.permute(0, 3, 1, 2)
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
        return x


class DinoV2Encoder(torch.nn.Module):
    def __init__(self, model_name="dinov2-base", freeze=True, device="cuda"):
        super().__init__()
        assert model_name in [
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
        ]
        self.device = device

        model_name = f"facebook/{model_name}"
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

        from vip import load_vip

        self.device = device
        self.model = load_vip().module.to(device)
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
    def __init__(
        self, model_name="r3m_18", freeze=True, device="cuda", *args, **kwargs
    ) -> None:
        super().__init__()
        from r3m import load_r3m

        assert model_name in ("r3m_18", "r3m_34", "r3m_50")

        model_name = f"resnet{model_name.split('_')[1]}"

        self.device = device
        self.model = load_r3m(modelid=model_name, device=device).module.to(device)
        self.encoding_dim = dict(
            resnet18=512,
            resnet34=512,
            resnet50=2048,
        )[model_name]

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


class DinoEncoder(torch.nn.Module):
    def __init__(self, freeze=True, device="cuda", *args, **kwargs) -> None:
        super().__init__()
        self.device = device

        # Model wants a batch of images of shape (batch_size, 3, 224, 224) and normalized
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")

        self.encoding_dim = self.model.norm.normalized_shape[0]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

        self.model = self.model.to(device)

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Expect input to be a batch of images of shape (batch_size, 224, 224, 3) in range [0, 255]
    def forward(self, x):
        # Move channels to the front
        x = x.permute(0, 3, 1, 2)
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
        return x


class MAEEncoder(torch.nn.Module):
    def __init__(self, freeze=True, device="cuda", *args, **kwargs) -> None:
        super().__init__()
        self.device = device

        wts = "/data/pulkitag/models/ankile/furniture-diffusion/mae/mae_pretrain_vit_base.pth"

        # Model wants a batch of images of shape (batch_size, 3, 224, 224) and normalized
        vit = vit_base_patch16()
        state_dict = torch.load(wts, map_location=device)
        vit.load_state_dict(state_dict["model"], strict=False)

        self.model = vit

        self.encoding_dim = 768

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.model = self.model.to(device)

    # Expect input to be a batch of images of shape (batch_size, 224, 224, 3) in range [0, 255]
    def forward(self, x):
        # Move channels to the front
        x = x.permute(0, 3, 1, 2)
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model.forward_features(x)
        return x


class VoltronEncoder(torch.nn.Module):
    def __init__(self, freeze=True, device="cuda", *args, **kwargs) -> None:
        super().__init__()

        from voltron import instantiate_extractor, load as load_voltron

        # Load a frozen Voltron (V-Cond) model & configure a vector extractor
        vcond, preprocess = load_voltron(
            "v-cond",
            device=device,
            freeze=freeze,
            cache="/data/scratch/ankile/.voltron",
        )
        vector_extractor = instantiate_extractor(vcond)().to(device)

        self.model = vcond
        self.preprocess = preprocess
        self.vector_extractor = vector_extractor

        self.encoding_dim = 384

    def forward(self, x, lang=None):
        x = x.permute(0, 3, 1, 2)
        x = self.preprocess(x)
        if lang is not None:
            x = self.model(x, lang, mode="multimodal")
        else:
            x = self.model(x, mode="visual")

        x = self.vector_extractor(x)
        return x
