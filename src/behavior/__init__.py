from omegaconf import DictConfig

from src.behavior.base import Actor
from src.dataset.normalizer import Normalizer


def get_actor(cfg: DictConfig, normalizer: Normalizer, device) -> Actor:
    """Returns an actor model."""
    actor_name = cfg.actor.name
    obs_type = cfg.observation_type

    assert obs_type in ["image", "state"], f"Invalid observation type: {obs_type}"
    assert (
        obs_type == "image" or actor_name == "mlp"
    ), "Only MLP actor supports state observation"

    if actor_name == "mlp":

        if obs_type == "image":
            from src.behavior.mlp import MLPActor

            return MLPActor(
                device=device,
                encoder_name=cfg.vision_encoder.model,
                freeze_encoder=cfg.vision_encoder.freeze,
                normalizer=normalizer,
                config=cfg,
            )

        if obs_type == "state":
            from src.behavior.mlp import MLPStateActor

            return MLPStateActor(
                device=device,
                normalizer=normalizer,
                config=cfg,
            )

    elif actor_name == "rnn":
        from src.behavior.rnn import RNNActor

        return RNNActor(
            device=device,
            encoder_name=cfg.vision_encoder.model,
            freeze_encoder=cfg.vision_encoder.freeze,
            normalizer=normalizer,
            config=cfg,
        )
    elif actor_name == "diffusion":
        assert not (
            cfg.multitask.get("multitask", False)
            and cfg.get("success_guidance", {}).get("success_guidance", False)
        ), "Multitask and success guidance cannot be used together"

        if cfg.multitask.multitask:
            from src.behavior.diffusion import MultiTaskDiffusionPolicy

            return MultiTaskDiffusionPolicy(
                device=device,
                encoder_name=cfg.vision_encoder.model,
                freeze_encoder=cfg.vision_encoder.freeze,
                normalizer=normalizer,
                config=cfg,
            )
        else:
            from src.behavior.diffusion import DiffusionPolicy

            return DiffusionPolicy(
                device=device,
                encoder_name=cfg.vision_encoder.model,
                freeze_encoder=cfg.vision_encoder.freeze,
                normalizer=normalizer,
                config=cfg,
            )
    elif actor_name == "guided_diffusion":
        from src.behavior.diffusion import SuccessGuidedDiffusionPolicy

        return SuccessGuidedDiffusionPolicy(
            device=device,
            encoder_name=cfg.vision_encoder.model,
            freeze_encoder=cfg.vision_encoder.freeze,
            normalizer=normalizer,
            config=cfg,
        )
    raise ValueError(f"Unknown actor type: {cfg.actor}")
