from ml_collections import ConfigDict

from src.behavior.base import Actor
from src.behavior.mlp import MLPActor
from src.behavior.rnn import RNNActor
from src.behavior.diffusion import DiffusionPolicy, MultiTaskDiffusionPolicy

from src.dataset.normalizer import StateActionNormalizer


def get_actor(config: ConfigDict, device) -> Actor:
    """Returns an actor model."""
    if config.actor == "mlp":
        return MLPActor(
            device=device,
            encoder_name=config.vision_encoder.model,
            freeze_encoder=config.vision_encoder.freeze,
            normalizer=StateActionNormalizer(),
            config=config,
        )
    elif config.actor == "rnn":
        return RNNActor(
            device=device,
            encoder_name=config.vision_encoder.model,
            freeze_encoder=config.vision_encoder.freeze,
            normalizer=StateActionNormalizer(),
            config=config,
        )
    elif config.actor == "diffusion":
        if "multi_task" in config and config.multi_task:
            return MultiTaskDiffusionPolicy(
                device=device,
                encoder_name=config.vision_encoder.model,
                freeze_encoder=config.vision_encoder.freeze,
                normalizer=StateActionNormalizer(act_rot_repr=config.act_rot_repr),
                config=config,
            )
        else:
            return DiffusionPolicy(
                device=device,
                encoder_name=config.vision_encoder.model,
                freeze_encoder=config.vision_encoder.freeze,
                normalizer=StateActionNormalizer(act_rot_repr=config.act_rot_repr),
                config=config,
            )

    raise ValueError(f"Unknown actor type: {config.actor}")
