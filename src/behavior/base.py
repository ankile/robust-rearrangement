from abc import ABC
from collections import deque
import torch
import torch.nn as nn
from src.data.normalizer import StateActionNormalizer
from src.models.vision import get_encoder
from src.models.unet import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.models.value import DoubleCritic, ValueNetwork

from ipdb import set_trace as bp  # noqa
from typing import Union


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call BaseClass()"""
        print(f"{__class__.__name__}.__call__({args}, {kwargs})")
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class Actor(ABC, torch.nn.Module, metaclass=PostInitCaller):
    obs_horizon: int
    action_horizon: int

    def action(self, obs: deque):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def __post_init__(self, *args, **kwargs):
        raise NotImplementedError
