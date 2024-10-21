import mujoco
import dexhub

# from loop_rate_limiters import RateLimiter
import mink
import numpy as np
from ipdb import set_trace as bp


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

    def __init__(self, path, visualize=False):

        self.traj = dexhub.load(path)

        bp()

        self.model = dexhub.get_sim(self.traj)
        self.data = mujoco.MjData(self.model)

        if visualize:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=True, show_right_ui=True
            )
        else:
            self.viewer = None

        # rate = RateLimiter(500)

        self.data.qpos = self.traj.data[0].obs.mj_qpos
        self.data.qvel = self.traj.data[0].obs.mj_qvel

        mujoco.mj_forward(self.model, self.data)

        config = mink.Configuration(self.model)
        self.fk_model = config.model
        self.fk_data = config.data

        # Create IK solver
        self.ik_solver = InverseKinematicsSolver(self.model)

    def step(self, l_ee, l_gripper, r_ee, r_gripper):

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


if __name__ == "__main__":
    env = DualFrankaEnv(
        path="/Users/larsankile/code/dexhub-api/my_data/place_plate/2024-10-09-22-03-54.dex",
        visualize=True,
    )

    for i in range(len(env.traj.data)):
        action_qpos = env.traj.data[i].act.mj_ctrl
        l_ee, r_ee = env.fk(action_qpos)
        env.step(l_ee, action_qpos[7], r_ee, action_qpos[15])
