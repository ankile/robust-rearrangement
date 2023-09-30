import furniture_bench
from furniture_bench.envs.observation import DEFAULT_STATE_OBS

import gym


def get_env(gpu_id, obs_type="state", furniture="one_leg"):
    if obs_type == "state":
        return gym.make(
            "FurnitureSim-v0",
            furniture=furniture,  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
            num_envs=1,  # Number of parallel environments.
            resize_img=True,  # If true, images are resized to 224 x 224.
            concat_robot_state=True,  # If true, robot state is concatenated to the observation.
            obs_keys=DEFAULT_STATE_OBS
            + ["color_image1", "color_image2"],  # Specifies the observation keys.
            headless=True,  # If true, simulation runs without GUI.
            compute_device_id=gpu_id,
            graphics_device_id=gpu_id,
            init_assembled=False,  # If true, the environment is initialized with assembled furniture.
            np_step_out=False,  # If true, env.step() returns Numpy arrays.
            channel_first=False,  # If true, images are returned in channel first format.
            randomness="low",  # Level of randomness in the environment [low | med | high].
            high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input=False,  # If true, the initial camera inputs are saved.
            record=False,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps=3000,  # Maximum number of steps per episode.
            act_rot_repr="quat",  # Representation of rotation for action space. Options are 'quat' and 'axis'.
        )

    elif obs_type == "feature":
        return gym.make(
            "FurnitureSimImageFeature-v0",
            furniture=furniture,  # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
            encoder_type="vip",
            include_raw_images=True,
            num_envs=1,  # Number of parallel environments.
            headless=True,  # If true, simulation runs without GUI.
            compute_device_id=gpu_id,
            graphics_device_id=gpu_id,
            init_assembled=False,  # If true, the environment is initialized with assembled furniture.
            randomness="low",  # Level of randomness in the environment [low | med | high].
            high_random_idx=-1,  # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input=False,  # If true, the initial camera inputs are saved.
            record=False,  # If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps=3000,  # Maximum number of steps per episode.
            act_rot_repr="quat",  # Representation of rotation for action space. Options are 'quat' and 'axis'.
        )
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")
