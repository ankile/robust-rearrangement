from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as bp
from src.models.module_attr_mixin import ModuleAttrMixin
from src.common.pytorch_util import replace_submodules
from src.models.vit import vit_base_patch16


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
        freeze=True,
        device="cuda",
        use_groupnorm=True,
        num_kp=32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        from robomimic.models.obs_core import VisualCore

        self.input_shape = [3, 224, 224]

        self.model = VisualCore(
            input_shape=self.input_shape,
            backbone_class="ResNet18Conv",
            pool_class="SpatialSoftmax",
            pool_kwargs={"num_kp": num_kp},
            flatten=True,
            feature_dimension=None,
        )

        self.encoding_dim = self.model.output_shape(self.input_shape)[0]

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

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
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
        super().__init__()
        assert model_name in ["resnet18", "resnet34", "resnet50"]
        assert not freeze or pretrained, "If not pretrained, then freeze must be False"
        print(f"Loading resnet, pretrained={pretrained}")

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

        if freeze:
            self.freeze()

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
        return x


class DinoV2Encoder(torch.nn.Module):
    def __init__(self, model_name="dinov2_vits14", freeze=True, device="cuda"):
        super().__init__()
        assert model_name in [
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ]
        self.device = device

        # Model wants a batch of images of shape (batch_size, 3, 224, 224) and normalized
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)

        self.encoding_dim = self.model.norm.normalized_shape[0]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

        self.model = self.model.to(device)

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
        return x


class ClipViTEncoder(torch.nn.Module):
    def __init__(self, model_name="vit_clip", freeze=True, device="cuda"):
        super().__init__()
        self.device = device

        import timm
        from timm.models.vision_transformer import VisionTransformer

        # Model wants a batch of images of shape (batch_size, 3, 224, 224) and normalized
        self.model: VisionTransformer = timm.create_model(
            model_name="vit_base_patch16_clip_224.openai",
            pretrained=True,
            global_pool="token",
            num_classes=0,
        )

        self.encoding_dim = self.model.embed_dim

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

        self.model = self.model.to(device)

        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.normalize = timm.data.create_transform(**data_cfg).transforms[-1]

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
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

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
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
        self.model = load_r3m(modelid=model_name).module.to(device)
        self.encoding_dim = dict(
            resnet18=512,
            resnet34=512,
            resnet50=2048,
        )[model_name]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
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

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
        return x


class MAEEncoder(torch.nn.Module):
    def __init__(self, freeze=True, device="cuda", *args, **kwargs) -> None:
        super().__init__()
        self.device = device

        # Get the home folder
        # Weights downloaded from: https://github.com/facebookresearch/mae?tab=readme-ov-file#fine-tuning-with-pre-trained-checkpoints
        wts = Path("~").expanduser() / ".mae" / "mae_pretrain_vit_base.pth"

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

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
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
            # cache="/data/scratch/ankile/.voltron",
        )
        vector_extractor = instantiate_extractor(vcond)().to(device)

        self.model = vcond
        self.preprocess = preprocess
        self.vector_extractor = vector_extractor

        self.encoding_dim = 384

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x, lang=None):
        x = self.preprocess(x)
        if lang is not None:
            x = self.model(x, lang, mode="multimodal")
        else:
            x = self.model(x, mode="visual")

        x = self.vector_extractor(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionPool2d(nn.Module):
    def __init__(
        self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spatial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor):
        bp()
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class DualInputAttentionPool2d(nn.Module):
    def __init__(
        self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(2 * spatial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1, x2 shape: (batch_size, embed_dim, spatial_dim, spatial_dim)

        # Reshape and concatenate inputs
        x1 = x1.flatten(start_dim=2).permute(2, 0, 1)  # (HW)NC
        x2 = x2.flatten(start_dim=2).permute(2, 0, 1)  # (HW)NC
        x = torch.cat([x1, x2], dim=0)  # (2*HW)NC

        # Add global token
        global_token = x.mean(dim=0, keepdim=True)  # 1NC
        x = torch.cat([global_token, x], dim=0)  # (2*HW+1)NC

        # Add positional embeddings
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (2*HW+1)NC

        # Apply multi-head attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class DualInputProprioceptionAttentionPool2d(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        embed_dim: int,
        num_heads: int,
        prop_dim: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(2 * spatial_dim**2 + 2, embed_dim) / embed_dim**0.5
        )
        self.prop_projection = nn.Linear(prop_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, prop: torch.Tensor):
        # x1, x2 shape: (batch_size, embed_dim, spatial_dim, spatial_dim)
        # prop shape: (batch_size, prop_dim)

        # Reshape and concatenate visual inputs
        x1 = x1.flatten(start_dim=2).permute(2, 0, 1)  # (HW)NC
        x2 = x2.flatten(start_dim=2).permute(2, 0, 1)  # (HW)NC
        x = torch.cat([x1, x2], dim=0)  # (2*HW)NC

        # Project proprioception to embed_dim
        prop_embedded = self.prop_projection(prop).unsqueeze(0)  # 1NC

        # Add global token
        global_token = x.mean(dim=0, keepdim=True)  # 1NC

        # Concatenate global token, proprioception, and visual features
        x = torch.cat([global_token, prop_embedded, x], dim=0)  # (2*HW+2)NC

        # Add positional embeddings
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (2*HW+2)NC

        # Apply multi-head attention
        x, _ = F.multi_head_attention_forward(
            query=x[:2],  # Use global token and proprioception as queries
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


if __name__ == "__main__":
    # Usage example
    spatial_dim = 7  # for 7x7 feature maps
    embed_dim = 512  # assuming 512-dimensional features from ResNet
    num_heads = 8
    output_dim = 256

    pool = DualInputAttentionPool2d(spatial_dim, embed_dim, num_heads, output_dim)
    resnet_output1 = torch.randn(
        32, embed_dim, spatial_dim, spatial_dim
    )  # Batch size of 32
    resnet_output2 = torch.randn(32, embed_dim, spatial_dim, spatial_dim)
    aggregated_features = pool(resnet_output1, resnet_output2)
    print(aggregated_features.shape)  # Should output: torch.Size([32, 128])


class TimmEncoder(nn.Module):
    """
    Creates a vision encoder with the timm library.

    It uses the forward features of the model to get the encoding.
    """

    def __init__(
        self,
        model_name,
        freeze=True,
        device="cuda",
        feature_aggregation="mean",
        num_attention_heads=8,
        embedding_dim=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.device = device
        self.feature_aggregation = feature_aggregation

        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        self.num_features = self.model.feature_info[-1]["num_chs"]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.model = self.model.to(device)

        data_cfg = timm.data.resolve_data_config(self.model.default_cfg)
        self.normalize = timm.data.create_transform(**data_cfg).transforms[-1]

        # Calculate spatial dimensions of the last feature map
        input_size = data_cfg["input_size"][1]  # Assuming square input
        total_stride = self.model.feature_info[-1]["reduction"]
        self.spatial_dim = input_size // total_stride

        # Define embedding_dim based on feature_aggregation method
        if embedding_dim is None:
            if feature_aggregation in ["mean", "max", "attention"]:
                self.embedding_dim = self.num_features
            elif feature_aggregation == "flatten":
                self.embedding_dim = (
                    self.num_features * self.spatial_dim * self.spatial_dim
                )
        else:
            self.embedding_dim = embedding_dim

        if feature_aggregation == "attention":
            self.attention_pool = AttentionPool2d(
                spacial_dim=self.spatial_dim,
                embed_dim=self.num_features,
                num_heads=num_attention_heads,
                output_dim=self.embedding_dim,
            )
        elif feature_aggregation in ["mean", "max"]:
            self.projection = nn.Linear(self.num_features, self.embedding_dim)
        elif feature_aggregation == "flatten":
            self.projection = nn.Linear(
                self.num_features * self.spatial_dim * self.spatial_dim,
                self.embedding_dim,
            )

    def aggregate_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_aggregation == "mean":
            x = x.mean(dim=[2, 3])
            return self.projection(x) if hasattr(self, "projection") else x
        elif self.feature_aggregation == "max":
            x = x.max(dim=[2, 3])[0]
            return self.projection(x) if hasattr(self, "projection") else x
        elif self.feature_aggregation == "attention":
            return self.attention_pool(x)
        elif self.feature_aggregation == "flatten":
            x = x.flatten(start_dim=1)
            return self.projection(x) if hasattr(self, "projection") else x
        else:
            raise ValueError(f"Invalid feature aggregation: {self.feature_aggregation}")

    def forward(self, x):
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        features = self.model(x)
        x = features[-1]  # Get the last feature map
        x = self.aggregate_features(x)
        return x
