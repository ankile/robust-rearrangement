import furniture_bench  # noqa: F401
from furniture_bench.envs.observation import DEFAULT_STATE_OBS, DEFAULT_VISUAL_OBS

import gym

from src.common.context import suppress_all_output


def get_env(
    gpu_id,
    furniture="one_leg",
    num_envs=1,
    randomness="low",
    resize_img=True,
    act_rot_repr="quat",
    ctrl_mode: str = "osc",
    verbose=False,
):
    with suppress_all_output(True):
        env = gym.make(
            "FurnitureSim-v0",
            furniture=furniture,  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
            num_envs=num_envs,  # Number of parallel environments.
            resize_img=resize_img,  # If true, images are resized to 224 x 224.
            concat_robot_state=True,  # If true, robot state is concatenated to the observation.
            headless=True,  # If true, simulation runs without GUI.
            obs_keys=DEFAULT_VISUAL_OBS,
            compute_device_id=gpu_id,
            graphics_device_id=gpu_id,
            init_assembled=False,  # If true, the environment is initialized with assembled furniture.
            np_step_out=False,  # If true, env.step() returns Numpy arrays.
            channel_first=False,  # If true, images are returned in channel first format.
            randomness=randomness,  # Level of randomness in the environment [low | med | high].
            high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input=False,  # If true, the initial camera inputs are saved.
            record=False,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps=3000,  # Maximum number of steps per episode.
            act_rot_repr=act_rot_repr,  # Representation of rotation for action space. Options are 'quat' and 'axis'.
            ctrl_mode=ctrl_mode,  # Control mode for the robot. Options are 'osc' and 'diffik'.
            verbose=verbose,  # If true, prints debug information.
        )

    return env
