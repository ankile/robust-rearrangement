import torch.nn as nn
from omegaconf import DictConfig

from src.models.unet import ConditionalUnet1D
from src.models.transformer import DiffusionTransformer
from src.models.vision import VisionEncoder

from ipdb import set_trace as bp


def get_encoder(
    encoder_name,
    device="cuda",
    freeze=True,
    pretrained=True,
    *args,
    **kwargs,
) -> VisionEncoder:
    if encoder_name.startswith("dinov2"):
        from src.models.vision import DinoV2Encoder

        return DinoV2Encoder(model_name=encoder_name, freeze=freeze, device=device)

    if encoder_name.startswith("clip"):
        from src.models.vision import ClipViTEncoder

        return ClipViTEncoder(model_name=encoder_name, freeze=freeze, device=device)
    if encoder_name.startswith("r3m"):
        from src.models.vision import R3MEncoder

        return R3MEncoder(model_name=encoder_name, freeze=freeze, device=device)
    if encoder_name == "vip":
        from src.models.vision import VIPEncoder

        return VIPEncoder(freeze=freeze, device=device)
    if encoder_name.startswith("resnet"):
        from src.models.vision import ResnetEncoder

        return ResnetEncoder(
            model_name=encoder_name,
            device=device,
            freeze=freeze,
            pretrained=pretrained,
            *args,
            **kwargs,
        )
    if encoder_name == "spatial_softmax":
        from src.models.vision import SpatialSoftmaxEncoder

        return SpatialSoftmaxEncoder(device=device, *args, **kwargs)
    if encoder_name == "dino":
        from src.models.vision import DinoEncoder

        return DinoEncoder(freeze=freeze, device=device)

    if encoder_name == "mae":
        from src.models.vision import MAEEncoder

        return MAEEncoder(freeze=freeze, device=device)
    if encoder_name == "voltron":
        from src.models.vision import VoltronEncoder

        return VoltronEncoder(freeze=freeze, device=device)
    raise ValueError(f"Unknown encoder name: {encoder_name}")


def get_diffusion_backbone(
    action_dim: int, obs_dim: int, actor_config: DictConfig
) -> nn.Module:
    if actor_config.diffusion_model.name == "unet":
        return ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            down_dims=actor_config.diffusion_model.down_dims,
        )
    elif actor_config.diffusion_model.name == "transformer":
        return DiffusionTransformer(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=actor_config.pred_horizon,
            n_obs_steps=actor_config.obs_horizon,
            cond_dim=obs_dim,
            n_layer=actor_config.diffusion_model.n_layer,
            n_head=actor_config.diffusion_model.n_head,
            n_emb=actor_config.diffusion_model.n_emb,
            p_drop_emb=actor_config.diffusion_model.p_drop_emb,
            p_drop_attn=actor_config.diffusion_model.p_drop_attn,
            causal_attn=actor_config.diffusion_model.causal_attn,
            time_as_cond=actor_config.diffusion_model.time_as_cond,
            obs_as_cond=actor_config.diffusion_model.obs_as_cond,
            n_cond_layers=actor_config.diffusion_model.n_cond_layers,
        )
    else:
        raise ValueError(f"Backbone {actor_config.model.backbone} not supported")
