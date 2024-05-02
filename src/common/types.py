import numpy as np
from typing import List, Tuple, TypedDict, Literal


class Observation(TypedDict):
    color_image1: np.ndarray
    color_image2: np.ndarray
    robot_state: dict
    image_size: Tuple[int]
    parts_poses: np.ndarray


class Trajectory(TypedDict):
    observations: List[Observation]
    actions: List[np.ndarray]
    rewards: List[float]
    skills: List[str]
    success: bool
    furniture: str
    error: bool
    error_description: str


# Make type for the encoder name choices
EncoderName = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "spatial_softmax",
    "dino",
    "mae",
    "voltron",
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
    "vip",
    "r3m_18",
    "r3m_34",
    "r3m_50",
]

TaskName = Literal[
    "one_leg",
    "lamp",
    "round_table",
    "desk",
    "square_table",
    "cabinet",
    "chair",
    "stool",
]


Controllers = Literal["sim", "real"]

Domains = Literal["sim", "real"]

DemoSources = Literal["scripted", "rollout", "teleop", "augmentation"]

Randomness = Literal["low", "med", "high"]

DemoStatus = Literal["success", "failure", "partial_success"]
