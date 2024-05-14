from omegaconf import DictConfig

from src.behavior.base import Actor


def get_actor(cfg: DictConfig, device) -> Actor:
    """Returns an actor model."""
    actor_name = cfg.actor.name
    obs_type = cfg.observation_type

    assert obs_type in ["image", "state"], f"Invalid observation type: {obs_type}"

    if actor_name == "mlp":

        from src.behavior.mlp import MLPActor

        return MLPActor(
            device=device,
            config=cfg,
        )

    elif actor_name == "rnn":
        assert False, "RNN actor is not supported at the moment."
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
            assert False, "Multitask diffusion actor is not supported at the moment."
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
                config=cfg,
            )
    elif actor_name == "guided_diffusion":
        assert False, "Guided diffusion actor is not supported at the moment."
        from src.behavior.diffusion import SuccessGuidedDiffusionPolicy

        return SuccessGuidedDiffusionPolicy(
            device=device,
            encoder_name=cfg.vision_encoder.model,
            freeze_encoder=cfg.vision_encoder.freeze,
            normalizer=normalizer,
            config=cfg,
        )
    raise ValueError(f"Unknown actor type: {cfg.actor}")
