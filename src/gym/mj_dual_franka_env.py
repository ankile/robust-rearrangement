from collections import OrderedDict
import time
from typing import Dict, Tuple
import gymnasium as gym
import torch
from dart_physics.runs import load_robot_cfg
from dart_physics.utils.scene_gen import construct_scene
import mujoco
from mujoco import viewer
from src.common import geometry as C

from loop_rate_limiters import RateLimiter

from dart_physics import mink
import numpy as np
from ipdb import set_trace as bp

from dart_physics.cfgs.bimanual_insertion import task_cfg, reset_function
from src.common.files import get_processed_path


class InverseKinematicsSolver:
    def __init__(self, model):
        self.config = mink.Configuration(model)

        # Create frame tasks for both end-effectors
        self.l_ee_task = mink.tasks.FrameTask(
            "l_robot/attachment", "body", position_cost=1, orientation_cost=1
        )
        self.r_ee_task = mink.tasks.FrameTask(
            "r_robot/attachment", "body", position_cost=1, orientation_cost=1
        )

        self.tasks = [self.l_ee_task, self.r_ee_task]

        # Add a PostureTask
        self.posture_task = mink.tasks.PostureTask(model, cost=0.01)

        target = self.config.q.copy()
        target[:9] = np.array([0, 0, 0, -1.57, 0, 1.57, -0.7853, 0.04, 0.04])
        target[9:18] = np.array([0, 0, 0, -1.57, 0, 1.57, -0.7853, 0.04, 0.04])
        self.posture_task.set_target(target)
        self.tasks.append(self.posture_task)
        self.limits = [
            # mink.ConfigurationLimit(
            #     model=model, gain=0.99, min_distance_from_limits=0.01
            # )
        ]

    def solve(self, current_qpos, l_ee_target, r_ee_target):
        """
        Solve inverse kinematics for both end-effectors.

        :param current_qpos: Current robot joint positions
        :param l_ee_target: Target pose for left end-effector (4x4 matrix)
        :param r_ee_target: Target pose for right end-effector (4x4 matrix)
        :return: New joint positions
        """
        # Set target poses
        self.l_ee_task.set_target(mink.SE3.from_matrix(l_ee_target))
        self.r_ee_task.set_target(mink.SE3.from_matrix(r_ee_target))

        # Update configuration
        self.config.update(current_qpos)

        # Solve IK
        dt = 0.002  # Integration timestep
        q_vel = mink.solve_ik(
            self.config, self.tasks, dt, solver="quadprog", limits=self.limits
        )

        q_target = self.config.integrate(q_vel, dt)

        return q_target


class DualFrankaEnv(gym.Env):

    viewer = None
    rate_limiter = None

    def __init__(
        self,
        concat_robot_state=True,
        observation_type="state",
        device="cpu",
        visualize=False,
    ):
        super().__init__()

        self.robot = "dual_panda"
        self.robot_cfg = load_robot_cfg(self.robot)
        self.task_cfg = task_cfg
        self.observation_type = observation_type

        # TODO: Check if this is correct
        self.model = construct_scene(self.task_cfg, self.robot_cfg)[0]

        self.data = mujoco.MjData(self.model)

        if visualize:
            self.viewer = viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=True, show_right_ui=True
            )
            self.rate_limiter = RateLimiter(50, warn=False)

        mujoco.mj_forward(self.model, self.data)

        config = mink.Configuration(self.model)
        self.fk_model = config.model
        self.fk_data = config.data

        if concat_robot_state:
            robot_state_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
            )
        else:
            robot_state_space = gym.spaces.Dict(
                OrderedDict(
                    {
                        "l_pos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                        ),
                        "l_rot": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "l_vel": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "l_gripper_width": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                        "r_pos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                        ),
                        "r_rot": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "r_vel": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "r_gripper_width": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                    }
                )
            )

        observation_space_dict = OrderedDict(
            {
                "robot_state": robot_state_space,
                "parts_poses": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
                ),
            }
        )

        if self.observation_type == "image":
            observation_space_dict["color_image1"] = gym.spaces.Box(
                low=0, high=255, shape=(1,), dtype=np.uint8
            )
            observation_space_dict["color_image2"] = gym.spaces.Box(
                low=0, high=255, shape=(720, 1280, 3), dtype=np.uint8
            )

        self.observation_space = gym.spaces.Dict(observation_space_dict)

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(20,), dtype=np.float32
        )

        # Create IK solver
        self.ik_solver = InverseKinematicsSolver(self.model)

        self.task_name = "bimanual_insertion"
        self.n_parts_assemble = 1

        self.pegname = "bimanual_insertion_peg/bimanual_insertion_peg"
        self.holename = "bimanual_insertion_hole3/bimanual_insertion_hole3"

        self.concat_robot_state = concat_robot_state
        self.device = device

        self.goal_pose = np.array(
            [
                [9.99942791e-01, 2.77529438e-04, 1.06928881e-02, -2.52389660e-04],
                [-1.94699848e-04, 9.99969977e-01, -7.74649435e-03, 3.11941604e-05],
                [-1.06947169e-02, 7.74396927e-03, 9.99912823e-01, -2.90148800e-02],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        # self.goal_pose = np.array(
        #     [
        #         [9.99942791e-01, 2.77529438e-04, 1.06928881e-02, 0],
        #         [-1.94699848e-04, 9.99969977e-01, -7.74649435e-03, 0],
        #         [-1.06947169e-02, 7.74396927e-03, 9.99912823e-01, -1.5e-02],
        #         [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        #     ]
        # )
        init_poses_path = get_processed_path(
            domain="sim",
            controller="dexhub",
            task="bimanual_insertion",
            demo_outcome="success",
            demo_source="teleop",
            randomness="low",
        ).parent
        self.init_poses = np.load(init_poses_path / "init_poses.npy")

        self.success = False

        self.camera_height = 720
        self.camera_width = 1280

        self.success_steps = 0

    def step(self, action: np.ndarray, sample_perturbations=False):
        assert sample_perturbations is False

        l_mat = self.learnable_to_mat(action[0:9])
        r_mat = self.learnable_to_mat(action[10:19])

        # Solve inverse kinematics
        q_new = self.ik_solver.solve(self.data.qpos, l_mat, r_mat)

        self.data.ctrl[:7] = q_new[:7]
        self.data.ctrl[8:15] = q_new[9:16]

        self.data.ctrl[7] = action[9]
        self.data.ctrl[15] = action[19]

        mujoco.mj_step(self.model, self.data, nstep=10)

        if self.viewer is not None:
            self.viewer.sync()
            self.rate_limiter.sleep()

        obs = self.get_observation()
        self.success |= self.is_success()
        reward = self.success

        return obs, reward, self.success, False, {}

    def learnable_to_mat(self, learnable: np.ndarray) -> np.ndarray:
        # Convert the action to 4x4 matrices
        mat = np.eye(4)

        # Add the translation
        mat[:3, 3] = learnable[:3]

        # Add the rotation
        mat[:3, :3] = C.np_rotation_6d_to_matrix(learnable[3:9])

        return mat

    def get_robot_state(self):
        qpos = self.data.qpos

        l_pos, l_rot, l_vel = self.body_to_learnable("l_robot/attachment")
        r_pos, r_rot, r_vel = self.body_to_learnable("r_robot/attachment")

        l_gripper_width = np.array([sum(qpos[7:9])])
        r_gripper_width = np.array([sum(qpos[16:18])])

        robot_state = {
            "l_pos": l_pos,
            "l_rot": l_rot,
            "l_vel": l_vel,
            "l_gripper_width": l_gripper_width,
            "r_pos": r_pos,
            "r_rot": r_rot,
            "r_vel": r_vel,
            "r_gripper_width": r_gripper_width,
        }

        # Combine all states
        if self.concat_robot_state:
            robot_state = np.concatenate(list(robot_state.values()), axis=-1)

        return robot_state

    def body_to_pose_and_vel(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        frame = self.data.body(body_name)
        pose_matrix = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(frame.xquat), translation=frame.xpos
        ).as_matrix()

        return pose_matrix, frame.cvel

    def mat_to_learnable(self, mat: np.ndarray) -> Tuple[np.ndarray]:
        """
        Convert a 4x4 matrix to a learnable representation.
        """
        pos = mat[:3, 3]
        rot = C.np_matrix_to_rotation_6d(mat[:3, :3])

        return pos, rot

    def body_to_learnable(
        self, body_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a body's pose and velocity to a learnable representation.
        """
        pose_matrix, vel = self.body_to_pose_and_vel(body_name)
        pos, rot = self.mat_to_learnable(pose_matrix)

        return pos, rot, vel

    def get_parts_poses(self):
        # Get the parts poses
        peg_pos, peg_quat_xyzw = self.data.body(
            self.pegname
        ).xpos, C.np_quat_wxyz_to_xyzw(self.data.body(self.pegname).xquat)
        hole_pos, hole_quat_xyzw = self.data.body(
            self.holename
        ).xpos, C.np_quat_wxyz_to_xyzw(self.data.body(self.holename).xquat)

        peg_pose = np.concatenate([peg_pos, peg_quat_xyzw])
        hole_pose = np.concatenate([hole_pos, hole_quat_xyzw])
        parts_poses = np.concatenate([peg_pose, hole_pose], axis=-1)

        return parts_poses

    def get_observation(self):
        obs = {
            "robot_state": self.get_robot_state(),
            "parts_poses": self.get_parts_poses(),
        }

        if self.observation_type == "image":
            obs["color_image1"] = np.array([0], dtype=np.uint8)
            obs["color_image2"] = self.render_camera()

        return obs

    def compute_reward(self):
        return float(self.is_success())

    def is_success(self, pos_tolerance=0.005, rot_tolerance_rad=0.05):
        """
        Check if the relative pose between peg and hole is close enough to goal pose.

        Args:
            pos_tolerance: Maximum allowed positional error in meters
            rot_tolerance_rad: Maximum allowed rotational error in radians

        Returns:
            success: Boolean indicating if pose is within tolerances
            metrics: Dict with detailed error metrics
        """
        # Get current transforms
        peg_mat = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(self.data.body(self.pegname).xquat),
            translation=self.data.body(self.pegname).xpos,
        ).as_matrix()
        hole_mat = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(self.data.body(self.holename).xquat),
            translation=self.data.body(self.holename).xpos,
        ).as_matrix()

        # Current relative pose
        curr_pose = peg_mat @ np.linalg.inv(hole_mat)

        # Compute relative error transform
        error_transform = np.linalg.inv(self.goal_pose) @ curr_pose

        # Extract position error
        pos_error = np.linalg.norm(error_transform[:3, 3])

        # Extract rotation error using matrix logarithm
        rot_mat_error = error_transform[:3, :3]
        rot_error_rad = np.arccos((np.trace(rot_mat_error) - 1) / 2)

        # Check if within tolerances
        pos_success = pos_error < pos_tolerance
        rot_success = rot_error_rad < rot_tolerance_rad
        success = pos_success and rot_success

        if self.success_steps < 1 and success:
            self.success_steps = 1

        if self.success_steps > 0:
            self.success_steps += 1

            if self.success_steps > 25:
                return True

        return False

    def exploding(self, threshold=5):

        # print max data.qvel
        if np.any(np.abs(self.data.qvel) > threshold):
            print("Max qvel", np.max(np.abs(self.data.qvel)))
            return True

        # Check if any nan or inf values in qpos or qvel
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            print("Nan values in qpos or qvel")
            return True

        if np.any(np.isinf(self.data.qpos)) or np.any(np.isinf(self.data.qvel)):
            print("Inf values in qpos or qvel")
            return True

        return False

    def render_camera(self):
        """Render the main_front camera view."""
        camera = self.model.camera("main_front")

        # Initialize camera configuration
        renderer = mujoco.Renderer(self.model, self.camera_height, self.camera_width)

        # Update scene and render
        renderer.update_scene(self.data, camera=camera.id)
        img = renderer.render()

        return img

    def reset(self):

        self.success = False
        self.success_steps = 0

        max_retries = 100

        for _ in range(max_retries):
            init_pose = self.init_poses[np.random.randint(len(self.init_poses))]

            self.data = reset_function(
                self.model, self.data, self.robot_cfg, self.task_cfg
            )

            self.data.qpos[:18] = init_pose
            self.data.qvel[:16] = 0.0

            for _ in range(5):
                mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

            if not self.exploding(threshold=2):
                break

        else:
            raise ValueError("Failed to reset the environment")

        if self.viewer is not None:
            self.viewer.sync()

        return self.get_observation(), {}


class DualFrankaVecEnv(gym.Env):
    def __init__(
        self,
        num_envs=1,
        concat_robot_state=True,
        observation_type="state",
        device="cpu",
        visualize=False,
    ) -> None:
        super().__init__()

        assert not visualize or num_envs == 1

        from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

        env_func = lambda: DualFrankaEnv(
            concat_robot_state=concat_robot_state,
            observation_type=observation_type,
            device=device,
            visualize=visualize,
        )
        VectorEnv = SyncVectorEnv if num_envs == 1 else AsyncVectorEnv
        self.envs = VectorEnv([env_func for _ in range(num_envs)])

        dummy_env = DualFrankaEnv(
            concat_robot_state=concat_robot_state, device=device, visualize=False
        )

        self.task_name = dummy_env.task_name
        self.n_parts_assemble = dummy_env.n_parts_assemble

        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space

        self.num_envs = num_envs
        self.device = device
        self.env_steps = torch.zeros(num_envs, device=device, dtype=torch.int32)

    def step(self, actions: torch.Tensor, sample_perturbations=False):
        assert sample_perturbations is False

        actions = actions.cpu().numpy()
        obs, rewards, dones, truncations, infos = self.envs.step(actions)

        # Convert to torch tensors
        obs = self.torchify(obs)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).bool().to(self.device)

        self.env_steps += 1

        return obs, rewards, dones, infos

    def reset(self):
        obs, _ = self.envs.reset()
        obs = self.torchify(obs)

        self.env_steps[:] = 0

        return obs

    def torchify(self, data):
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                data[key] = torch.from_numpy(val).float().to(self.device)
            elif isinstance(val, dict):
                data[key] = self.torchify(val)
            else:
                raise ValueError(f"Unsupported data type: {type(val)}")

        return data

    def filter_and_concat_robot_state(self, robot_state: Dict[str, torch.Tensor]):
        current_robot_state = [val for val in robot_state.values()]
        return torch.cat(current_robot_state, dim=-1)
