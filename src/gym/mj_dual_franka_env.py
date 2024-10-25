from collections import OrderedDict
from typing import Dict
import gymnasium as gym
import torch
from dart_physics.runs import load_robot_cfg
from dart_physics.utils.scene_gen import construct_scene
import mujoco
from mujoco import viewer
from src.common import geometry as C

from loop_rate_limiters import RateLimiter

import mink
import numpy as np
from ipdb import set_trace as bp

from dart_physics.cfgs.bimanual_insertion import task_cfg, reset_function
from src.common.files import get_processed_path


def custom_warning_callback(message, *args):
    pass


# Disable logging warnings
import logging

logging.getLogger().setLevel(logging.CRITICAL)


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
        device="cpu",
        visualize=False,
    ):
        super().__init__()

        self.robot = "dual_panda"
        self.robot_cfg = load_robot_cfg(self.robot)
        self.task_cfg = task_cfg

        # TODO: Check if this is correct
        self.model = construct_scene(self.task_cfg, self.robot_cfg)[0]

        self.data = mujoco.MjData(self.model)

        if visualize:
            self.viewer = viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=True, show_right_ui=True
            )
            self.rate_limiter = RateLimiter(50)

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
                        "l_pos_state": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                        ),
                        "l_rot_6d": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "l_vel": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "l_gripper_width": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                        "r_pos_state": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                        ),
                        "r_rot_6d": gym.spaces.Box(
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

        self.observation_space = gym.spaces.Dict(
            {
                "robot_state": robot_state_space,
                "parts_poses": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
                ),
            }
        )
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
        init_poses_path = get_processed_path(
            domain="sim",
            controller="dexhub",
            task="bimanual_insertion",
            demo_outcome="success",
            demo_source="teleop",
            randomness="low",
        ).parent
        self.init_poses = np.load(init_poses_path / "init_poses.npy")

    def step(self, action: np.ndarray, sample_perturbations=False):
        assert sample_perturbations is False

        l_ee, l_gripper, r_ee, r_gripper = (
            action[:9],
            action[9],
            action[10:19],
            action[19],
        )

        # Convert the action to 4x4 matrices
        l_mat, r_mat = np.eye(4), np.eye(4)

        # Add the translation
        l_mat[:3, 3], r_mat[:3, 3] = l_ee[:3], r_ee[:3]

        # Add the rotation
        l_mat[:3, :3], r_mat[:3, :3] = C.np_rotation_6d_to_matrix(
            l_ee[3:9]
        ), C.np_rotation_6d_to_matrix(r_ee[3:9])

        # Solve inverse kinematics
        q_new = self.ik_solver.solve(self.data.qpos, l_mat, r_mat)

        self.data.ctrl[:7] = q_new[:7]
        self.data.ctrl[8:15] = q_new[9:16]

        self.data.ctrl[7] = l_gripper
        self.data.ctrl[15] = r_gripper

        for _ in range(10):

            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()
            self.rate_limiter.sleep()

        obs = self.get_observation()

        reward = self.compute_reward()
        done = self.is_success()

        return obs, reward, done, False, {}

    def fk(self, ctrl):
        """
        Compute forward kinematics for the robot.
        """

        self.fk_data.qpos[:7] = ctrl[:7]
        self.fk_data.qpos[9:16] = ctrl[8:15]

        mujoco.mj_kinematics(self.fk_model, self.fk_data)

        l_frame = self.fk_data.body("l_robot/attachment")
        r_frame = self.fk_data.body("r_robot/attachment")

        l_ee = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(l_frame.xquat), translation=l_frame.xpos
        ).as_matrix()
        r_ee = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(r_frame.xquat), translation=r_frame.xpos
        ).as_matrix()

        # Get the velocities
        l_vel = self.fk_data.body("l_robot/attachment").cvel
        r_vel = self.fk_data.body("r_robot/attachment").cvel

        return l_ee, r_ee, l_vel, r_vel

    def get_robot_state(self):
        qpos = self.data.qpos

        l_frame = self.data.body("l_robot/attachment")
        r_frame = self.data.body("r_robot/attachment")

        l_ee = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(l_frame.xquat), translation=l_frame.xpos
        ).as_matrix()
        r_ee = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(r_frame.xquat), translation=r_frame.xpos
        ).as_matrix()

        # Get the velocities
        l_vel = l_frame.cvel
        r_vel = r_frame.cvel

        l_pos_state, r_pos_state = l_ee[:3, 3], r_ee[:3, 3]

        l_mat, r_mat = l_ee[:3, :3], r_ee[:3, :3]

        l_rot_6d, r_rot_6d = C.np_matrix_to_rotation_6d(
            l_mat
        ), C.np_matrix_to_rotation_6d(r_mat)

        l_gripper_width = qpos[7] + qpos[8]
        r_gripper_width = qpos[16] + qpos[17]

        robot_state = {
            "l_pos_state": l_pos_state,
            "l_rot_6d": l_rot_6d,
            "l_vel": l_vel,
            "l_gripper_width": np.array([l_gripper_width]),
            "r_pos_state": r_pos_state,
            "r_rot_6d": r_rot_6d,
            "r_vel": r_vel,
            "r_gripper_width": np.array([r_gripper_width]),
        }

        # Combine all states
        if self.concat_robot_state:
            robot_state = np.concatenate(list(robot_state.values()), axis=-1)

        return robot_state

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
        return obs

    def compute_reward(self):
        return float(self.is_success())

    def is_success(self):
        peg_mat = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(self.data.body(self.pegname).xquat),
            translation=self.data.body(self.pegname).xpos,
        ).as_matrix()
        hole_mat = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(self.data.body(self.holename).xquat),
            translation=self.data.body(self.holename).xpos,
        ).as_matrix()

        curr_pose = peg_mat @ np.linalg.inv(hole_mat)

        # Check if curr pose is close to goal pose
        success = np.allclose(curr_pose, self.goal_pose, atol=0.01)

        return success

    def reset(self):
        reset_function(self.model, self.data, self.robot_cfg, self.task_cfg)

        self.data.qpos[:18] = self.init_poses[np.random.randint(len(self.init_poses))]
        # self.data.qpos[:18] = self.init_poses[1]

        self.data.qvel = np.zeros_like(self.data.qvel)

        self.data.ctrl[:7] = self.data.qpos[:7]
        self.data.ctrl[8:15] = self.data.qpos[9:16]

        # Make sure the changes are reflected in the simulation
        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return self.get_observation(), {}


class DualFrankaVecEnv(gym.Env):
    def __init__(
        self,
        num_envs=1,
        concat_robot_state=True,
        device="cpu",
        visualize=False,
    ) -> None:
        super().__init__()

        assert not visualize or num_envs == 1

        from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

        env_func = lambda: DualFrankaEnv(
            concat_robot_state=concat_robot_state,
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
        obs, infos = self.envs.reset()
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
