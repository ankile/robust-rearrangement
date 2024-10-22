from typing import Dict
import gymnasium as gym
import torch
from dart_physics.runs import load_robot_cfg
from dart_physics.utils.scene_gen import construct_scene
import mujoco
from mujoco import viewer
import dexhub
from src.common import geometry as C

import mink
import numpy as np
from ipdb import set_trace as bp

from dart_physics.cfgs.bimanual_insertion import task_cfg, reset_function


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
        target[:9] = np.array(
            [
                0,
                0,
                0,
                -1.5707899999999999,
                0,
                1.5707899999999999,
                -0.7853,
                0.040000000000000001,
                0.040000000000000001,
            ]
        )
        target[9:18] = np.array(
            [
                0,
                0,
                0,
                -1.5707899999999999,
                0,
                1.5707899999999999,
                -0.7853,
                0.040000000000000001,
                0.040000000000000001,
            ]
        )
        self.posture_task.set_target(target)
        self.tasks.append(self.posture_task)
        self.limits = [
            # mink.ConfigurationLimit(
            #     model=model, gain=0.99, min_distance_from_limits=0.01
            # )
        ]

        # Add DampingTask
        # self.tasks.append(mink.tasks.DampingTask(model, cost=0.1))

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

    def __init__(
        self,
        concat_robot_state=True,
        device="cpu",
        visualize=False,
    ):

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
        else:
            self.viewer = None

        mujoco.mj_forward(self.model, self.data)

        config = mink.Configuration(self.model)
        self.fk_model = config.model
        self.fk_data = config.data

        # Create IK solver
        self.ik_solver = InverseKinematicsSolver(self.model)

        self.num_envs = 1
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

    def step(self, action, sample_perturbations=False):
        assert sample_perturbations is False

        action = action.squeeze().cpu().numpy()

        l_ee, l_gripper, r_ee, r_gripper = (
            action[:9],
            action[9],
            action[10:19],
            action[19],
        )

        # Map gripper action values from [-1, 1] to [0.04, 0.0]
        # l_gripper = 0.0 if l_gripper > 0.5 else 0.04
        # r_gripper = 0.0 if r_gripper > 0.5 else 0.04

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
            # rate.sleep()

        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.is_success()

        return obs, reward, done, {}

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

        l_gripper_width = qpos[8] - qpos[7]
        r_gripper_width = qpos[16] - qpos[15]

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

    def filter_and_concat_robot_state(self, robot_state: Dict[str, torch.Tensor]):
        current_robot_state = [val for val in robot_state.values()]
        return torch.cat(current_robot_state, dim=-1)

    def torchify(self, data):
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                data[key] = torch.from_numpy(val).unsqueeze(0).float().to(self.device)
            elif isinstance(val, dict):
                data[key] = self.torchify(val)

        return data

    def get_observation(self):
        obs = {
            "robot_state": self.get_robot_state(),
            "parts_poses": self.get_parts_poses(),
        }
        obs = self.torchify(obs)
        return obs

    def compute_reward(self):
        return self.is_success().float()

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

        return torch.tensor([success], device=self.device).unsqueeze(0)

    def reset(self):
        reset_function(self.model, self.data, self.robot_cfg, self.task_cfg)

        init_poses = np.load(
            "/data/scratch/ankile/robust-rearrangement/notebooks/init_poses.npy"
        )

        self.data.qpos[:18] = init_poses[np.random.randint(len(init_poses))]

        self.data.qvel = np.zeros_like(self.data.qvel)

        # mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)

        # Make sure the changes are reflected in the simulation
        mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return self.get_observation()
