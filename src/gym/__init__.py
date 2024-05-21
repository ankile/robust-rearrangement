from pathlib import Path
import furniture_bench  # noqa: F401
from furniture_bench.envs.observation import (
    DEFAULT_VISUAL_OBS,
    DEFAULT_STATE_OBS,
    FULL_OBS,
)
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

import gym

from src.common.context import suppress_all_output


def turn_off_april_tags():
    from furniture_bench.envs import furniture_sim_env

    furniture_sim_env.ASSET_ROOT = str(
        Path(__file__).parent.parent.absolute() / "assets"
    )


def get_env(
    gpu_id,
    furniture="one_leg",
    num_envs=1,
    randomness="low",
    max_env_steps=5_000,
    resize_img=True,
    observation_space="image",  # Observation space for the robot. Options are 'image' and 'state'.
    act_rot_repr="quat",
    ctrl_mode: str = "osc",
    action_type="delta",  # Action type for the robot. Options are 'delta' and 'pos'.
    april_tags=True,
    verbose=False,
    headless=True,
    **kwargs,
) -> FurnitureSimEnv:
    if not april_tags:
        from furniture_bench.envs import furniture_sim_env

        furniture_sim_env.ASSET_ROOT = str(
            Path(__file__).parent.parent.absolute() / "assets"
        )

    if observation_space == "image":
        obs_keys = DEFAULT_VISUAL_OBS + ["parts_poses"]
    elif observation_space == "state":
        obs_keys = DEFAULT_STATE_OBS
    else:
        raise ValueError("Invalid observation space")

    with suppress_all_output(not verbose):
        env = FurnitureSimEnv(
            furniture=furniture,  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
            num_envs=num_envs,  # Number of parallel environments.
            resize_img=resize_img,  # If true, images are resized to 224 x 224.
            concat_robot_state=True,  # If true, robot state is concatenated to the observation.
            headless=headless,  # If true, simulation runs without GUI.
            obs_keys=obs_keys,
            compute_device_id=gpu_id,
            graphics_device_id=gpu_id,
            init_assembled=False,  # If true, the environment is initialized with assembled furniture.
            np_step_out=False,  # If true, env.step() returns Numpy arrays.
            channel_first=False,  # If true, images are returned in channel first format.
            randomness=randomness,  # Level of randomness in the environment [low | med | high].
            high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input=False,  # If true, the initial camera inputs are saved.
            record=False,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps=max_env_steps,  # Maximum number of steps per episode.
            act_rot_repr=act_rot_repr,  # Representation of rotation for action space. Options are 'quat' and 'axis'.
            ctrl_mode=ctrl_mode,  # Control mode for the robot. Options are 'osc' and 'diffik'.
            action_type=action_type,  # Action type for the robot. Options are 'delta' and 'pos'.
            verbose=verbose,  # If true, prints debug information.
            **kwargs,
        )

    return env


from src.gym.furniture_sim_env import FurnitureRLSimEnv


def get_rl_env(
    gpu_id,
    furniture="one_leg",
    num_envs=1,
    randomness="low",
    max_env_steps=5_000,
    resize_img=True,
    observation_space="image",  # Observation space for the robot. Options are 'image' and 'state'.
    act_rot_repr="quat",
    ctrl_mode: str = "diffik",
    action_type="pos",  # Action type for the robot. Options are 'delta' and 'pos'.
    april_tags=False,
    verbose=False,
    headless=True,
    **kwargs,
) -> FurnitureRLSimEnv:
    if not april_tags:
        from furniture_bench.envs import furniture_sim_env

        furniture_sim_env.ASSET_ROOT = str(
            Path(__file__).parent.parent.absolute() / "assets"
        )

    # To ensure we can replay the rollouts, we need to (1) include all robot states in the observation space
    # and (2) ensure that the robot state is stored as a dict for compatibility with the teleop data
    obs_keys = FULL_OBS
    if observation_space == "state":
        # Filter out keys with `image` in them
        obs_keys = [key for key in obs_keys if "image" not in key]

    with suppress_all_output(not verbose):
        env = FurnitureRLSimEnv(
            furniture=furniture,  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
            num_envs=num_envs,  # Number of parallel environments.
            resize_img=resize_img,  # If true, images are resized to 224 x 224.
            concat_robot_state=False,  # If true, robot state is concatenated to the observation.
            headless=headless,  # If true, simulation runs without GUI.
            obs_keys=obs_keys,
            compute_device_id=gpu_id,
            graphics_device_id=gpu_id,
            init_assembled=False,  # If true, the environment is initialized with assembled furniture.
            np_step_out=False,  # If true, env.step() returns Numpy arrays.
            channel_first=False,  # If true, images are returned in channel first format.
            randomness=randomness,  # Level of randomness in the environment [low | med | high].
            high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input=False,  # If true, the initial camera inputs are saved.
            record=False,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps=max_env_steps,  # Maximum number of steps per episode.
            act_rot_repr=act_rot_repr,  # Representation of rotation for action space. Options are 'quat' and 'axis'.
            ctrl_mode=ctrl_mode,  # Control mode for the robot. Options are 'osc' and 'diffik'.
            action_type=action_type,  # Action type for the robot. Options are 'delta' and 'pos'.
            verbose=verbose,  # If true, prints debug information.
            **kwargs,
        )

    return env
