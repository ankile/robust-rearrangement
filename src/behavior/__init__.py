from omegaconf import DictConfig

from src.behavior.base import Actor
from src.behavior.mlp import MLPActor
from src.behavior.rnn import RNNActor
from src.behavior.diffusion import DiffusionPolicy, MultiTaskDiffusionPolicy


def get_actor(config: DictConfig, device) -> Actor:
    """Returns an actor model."""
    actor_name = config.actor.name
    if actor_name == "mlp":
        return MLPActor(
            device=device,
            encoder_name=config.vision_encoder.model,
            freeze_encoder=config.vision_encoder.freeze,
            config=config,
        )
    elif actor_name == "rnn":
        return RNNActor(
            device=device,
            encoder_name=config.vision_encoder.model,
            freeze_encoder=config.vision_encoder.freeze,
            config=config,
        )
    elif actor_name == "diffusion":
        if config.multitask.multitask:
            return MultiTaskDiffusionPolicy(
                device=device,
                encoder_name=config.vision_encoder.model,
                freeze_encoder=config.vision_encoder.freeze,
                config=config,
            )
        else:
            return DiffusionPolicy(
                device=device,
                encoder_name=config.vision_encoder.model,
                freeze_encoder=config.vision_encoder.freeze,
                config=config,
            )

    raise ValueError(f"Unknown actor type: {config.actor}")
