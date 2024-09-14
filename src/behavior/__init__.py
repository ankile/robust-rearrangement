from omegaconf import DictConfig

from src.behavior.base import Actor


def get_actor(cfg: DictConfig, device) -> Actor:
    """Returns an actor model."""
    actor_name = cfg.actor_name if "actor_name" in cfg else cfg.actor.name
    obs_type = cfg.observation_type

    assert obs_type in ["image", "state"], f"Invalid observation type: {obs_type}"

    if actor_name == "mlp":

        from src.behavior.mlp import MLPActor

        return MLPActor(
            cfg=cfg,
            device=device,
        )

    elif actor_name == "diffusion":
        from src.behavior.diffusion import DiffusionPolicy

        return DiffusionPolicy(
            cfg=cfg,
            device=device,
        )

    elif actor_name == "residual_diffusion":
        from src.behavior.residual_diffusion import ResidualDiffusionPolicy

        return ResidualDiffusionPolicy(
            cfg=cfg,
            device=device,
        )

    elif actor_name == "attentionpool_diffusion":
        from src.behavior.diffusion import AttentionPoolDiffusionPolicy

        return AttentionPoolDiffusionPolicy(
            cfg=cfg,
            device=device,
        )
    raise ValueError(f"Unknown actor type: {cfg.actor}")
