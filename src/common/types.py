from typing import List, Tuple, TypedDict


class Observation(TypedDict):
    color_image1: np.ndarray
    color_image2: np.ndarray
    robot_state: dict
    image_size: Tuple[int]


class Trajectory(TypedDict):
    observations: List[Observation]
    actions: List[np.ndarray]
    rewards: List[float]
    skills: List[str]
    success: bool
    furniture: str
    error: bool
    error_description: str
