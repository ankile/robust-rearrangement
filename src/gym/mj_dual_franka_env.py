from dart_physics.runs import load_robot_cfg
from dart_physics.utils.scene_gen import construct_scene
import mujoco
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

        # Add DampingTask
        self.tasks.append(mink.tasks.DampingTask(model, cost=0.01))

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
        q_vel = mink.solve_ik(self.config, self.tasks, dt, solver="quadprog")

        q_target = self.config.integrate(q_vel, dt)

        return q_target


class DualFrankaEnv:

    def __init__(self, path=None, visualize=False):

        path = "/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/dexhub/sim/bimanual_insertion"
        path += "/2024-10-09-22-03-25.dex"

        self.traj = dexhub.load(path)
        self.model = dexhub.get_sim(self.traj)

        self.robot = "dual_panda"
        # self.robot_cfg = load_robot_cfg(self.robot)
        self.robot_cfg = {"name": "DualPanda", "obj_startidx": (7 + 2) * 2}
        self.task_cfg = task_cfg

        # self.model = construct_scene(self.task_cfg, self.robot_cfg)

        self.data = mujoco.MjData(self.model)

        if visualize:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=True, show_right_ui=True
            )
        else:
            self.viewer = None

        self.data.qpos = self.traj.data[0].obs.mj_qpos
        self.data.qvel = self.traj.data[0].obs.mj_qvel

        mujoco.mj_forward(self.model, self.data)

        config = mink.Configuration(self.model)
        self.fk_model = config.model
        self.fk_data = config.data

        # Create IK solver
        self.ik_solver = InverseKinematicsSolver(self.model)

        self.num_envs = 1
        self.task_name = "bimanual_insertion"
        self.n_parts_assemble = 1

    def step(self, action):

        l_ee, l_gripper, r_ee, r_gripper = (
            action[:9],
            action[9],
            action[10:19],
            action[19],
        )

        # Solve inverse kinematics
        q_new = self.ik_solver.solve(self.data.qpos, l_ee, r_ee)

        self.data.ctrl[:7] = q_new[:7]
        self.data.ctrl[8:15] = q_new[9:16]

        self.data.ctrl[7] = l_gripper
        self.data.ctrl[15] = r_gripper

        for _ in range(10):

            mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()
            # rate.sleep()

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

        return l_ee, r_ee

    def get_robot_state(self):
        qpos = self.data.qpos

        l_ee, r_ee, l_vel, r_vel = self.fk(qpos)

        l_pos_state, r_pos_state = l_ee[:3, 3], r_ee[:3, 3]

        l_mat, r_mat = l_ee[:3, :3], r_ee[:3, :3]

        l_rot_6d, r_rot_6d = C.np_matrix_to_rotation_6d(
            l_mat
        ), C.np_matrix_to_rotation_6d(r_mat)

        l_gripper_width = qpos[8] - qpos[7]
        r_gripper_width = qpos[16] - qpos[15]

        # Combine all states
        robot_state = np.concatenate(
            [
                l_pos_state,
                l_rot_6d,
                l_vel,
                [l_gripper_width],
                r_pos_state,
                r_rot_6d,
                r_vel,
                [r_gripper_width],
            ]
        )

        return robot_state

    def get_parts_poses(self):
        # Get the parts poses
        peg_pos, peg_quat_xyzw = self.data.body(self.pegname).xpos, C.quat_wxyz_to_xyzw(
            self.data.body(self.pegname).xquat
        )
        hole_pos, hole_quat_xyzw = self.data.body(
            self.holename
        ).xpos, C.quat_wxyz_to_xyzw(self.data.body(self.holename).xquat)

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

    def reset(self):
        reset_function(self.model, self.data, self.robot_cfg, self.task_cfg)

        return self.get_observation()


if __name__ == "__main__":
    env = DualFrankaEnv(
        path="/Users/larsankile/code/dexhub-api/my_data/place_plate/2024-10-09-22-03-54.dex",
        visualize=True,
    )

    for i in range(len(env.traj.data)):
        action_qpos = env.traj.data[i].act.mj_ctrl
        l_ee, r_ee = env.fk(action_qpos)
        env.step(l_ee, action_qpos[7], r_ee, action_qpos[15])
