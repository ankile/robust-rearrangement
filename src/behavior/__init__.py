from omegaconf import DictConfig

from src.behavior.base import Actor
from src.dataset.normalizer import Normalizer


def get_actor(config: DictConfig, normalizer: Normalizer, device) -> Actor:
    """Returns an actor model."""
    actor_name = config.actor.name
    if actor_name == "mlp":
        from src.behavior.mlp import MLPActor

        return MLPActor(
            device=device,
            encoder_name=config.vision_encoder.model,
            freeze_encoder=config.vision_encoder.freeze,
            config=config,
        )
    elif actor_name == "rnn":
        from src.behavior.rnn import RNNActor

        return RNNActor(
            device=device,
            encoder_name=config.vision_encoder.model,
            freeze_encoder=config.vision_encoder.freeze,
            config=config,
        )
    elif actor_name == "diffusion":
        assert not (
            config.multitask.get("multitask", False)
            and config.get("success_guidance", {}).get("success_guidance", False)
        ), "Multitask and success guidance cannot be used together"

        if config.multitask.multitask:
            from src.behavior.diffusion import MultiTaskDiffusionPolicy

            return MultiTaskDiffusionPolicy(
                device=device,
                encoder_name=config.vision_encoder.model,
                freeze_encoder=config.vision_encoder.freeze,
                config=config,
            )
        else:
            from src.behavior.diffusion import DiffusionPolicy

            return DiffusionPolicy(
                device=device,
                encoder_name=config.vision_encoder.model,
                freeze_encoder=config.vision_encoder.freeze,
                normalizer=normalizer,
                config=config,
            )
    elif actor_name == "guided_diffusion":
        from src.behavior.diffusion import SuccessGuidedDiffusionPolicy

        return SuccessGuidedDiffusionPolicy(
            device=device,
            encoder_name=config.vision_encoder.model,
            freeze_encoder=config.vision_encoder.freeze,
            normalizer=normalizer,
            config=config,
        )
    raise ValueError(f"Unknown actor type: {config.actor}")
