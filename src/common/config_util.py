from ipdb import set_trace as bp
from omegaconf import DictConfig, OmegaConf


def merge_base_bc_config_with_root_config(cfg: DictConfig, base_cfg: DictConfig):
    """
    Mainly used in residual RL training, where a base BC model must be loaded and
    several of the base BC configs must be copied over to the root level config
    of our current run, as these keys are expected to exist + contain populated
    data in lower-level class constructors (e.g., Actor classes)

    Trying to make intentions clear that, depending on being in "state" mode or
    "image" mode, we need to copy over different keys
    """

    # put all things relating to base policy in "base_policy" key
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "base_policy", base_cfg, merge=True)

    # "actor" expected to have things populated that we only populate during BC training
    OmegaConf.update(cfg, "actor", base_cfg.actor, merge=True)

    # instantiating the residual policy with "base_cfg", we need to bring in the
    # residual policy INTO the base_cfg
    if "residual_policy" in cfg.actor:
        OmegaConf.update(
            base_cfg.actor, "residual_policy", cfg.actor.residual_policy, merge=True
        )

    if "critic" in cfg:
        OmegaConf.update(base_cfg.actor, "critic", cfg.critic, merge=True)
        OmegaConf.update(base_cfg.actor, "init_logstd", cfg.init_logstd, merge=True)

    # good to know what data the BC was trained on + some cfg.data expected in Actor
    OmegaConf.update(cfg, "data", base_cfg.data)

    # expected in Actor
    OmegaConf.update(cfg, "robot_state_dim", base_cfg.robot_state_dim)

    if cfg.observation_type == "image":
        # expected in Actor
        OmegaConf.update(cfg, "regularization", base_cfg.regularization)
    elif cfg.observation_type == "state":
        # expected in Actor
        OmegaConf.update(cfg, "parts_poses_dim", base_cfg.parts_poses_dim)

    # # Some final checks for full backward compatibility
    # if "control" not in cfg or cfg.control != base_cfg.control:
    #     OmegaConf.update(cfg, "control", base_cfg.control)
