from omegaconf import OmegaConf


def to_native(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj
