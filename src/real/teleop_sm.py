from pathlib import Path
import pickle
import time
from multiprocessing.managers import SharedMemoryManager
import torch
import numpy as np
from datetime import datetime
import meshcat
import scipy.spatial.transform as st
import pyrealsense2 as rs

from polymetis import GripperInterface

from rdt.spacemouse.spacemouse_shared_memory import Spacemouse

from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from rdt.image.factory import enable_single_realsense

from src.real.serials import FRONT_CAMERA_SERIAL, WRIST_CAMERA_SERIAL

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port_vis", type=int, default=6000)
parser.add_argument("--frequency", type=int, default=10)  # 30
parser.add_argument("--command_latency", type=float, default=0.01)
parser.add_argument("--deadzone", type=float, default=0.05)
parser.add_argument("--max_pos_speed", type=float, default=0.3)
parser.add_argument("--max_rot_speed", type=float, default=0.7)
parser.add_argument("--use_lcm", action="store_true")
parser.add_argument("--save_dir", required=True)
parser.add_argument("--furniture", type=str, required=True)

args = parser.parse_args()

poly_util = PolymetisHelper()


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


class ActionContainer:
    def __init__(
        self,
        next_pose_mat: np.ndarray,
        grasp_flag: int,
        action_taken: bool,
        collect_enum: CollectEnum,
        is_gripper_open: bool,
        toggle_gripper: bool,
    ):
        self.next_pose_mat = next_pose_mat
        self.grasp_flag = grasp_flag
        self.action_taken = action_taken
        self.collect_enum = collect_enum
        self.is_gripper_open = is_gripper_open
        self.toggle_gripper = toggle_gripper


# Setup observation and action helpers
class ObsActHelper:
    def __init__(
        self, sm, keyboard, robot, gripper, front_image_pipeline, wrist_image_pipeline
    ):

        self.sm = sm
        self.keyboard = keyboard

        self.robot = robot
        self.gripper = gripper

        self.front_image_pipeline = front_image_pipeline
        self.wrist_image_pipeline = wrist_image_pipeline

        self._setup()

    def set_constants(
        self,
        max_pos_speed: float,
        max_rot_speed: float,
        sm_dpos_scalar: float,
        sm_drot_scalar: float,
        frequency: float,
    ):
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.sm_dpos_scalar = sm_dpos_scalar
        self.sm_drot_scalar = sm_drot_scalar
        self.frequency = frequency

    def _setup(self):
        self.grasp_flag = -1
        self.gripper_open = True
        self.last_grip_step = 0
        self.steps_since_grasp = 0
        self.record_latency_when_grasping = 15

    @staticmethod
    def to_pose_mat(pose_):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = pose_[:3]
        pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix()
        return pose_mat

    def set_target_pose(self, target_pose: np.ndarray):
        # [x, y, z, dx, dy, dz] (rotvec!)
        self.target_pose = target_pose

    def get_action(self):
        # get teleop command
        sm_state = self.sm.get_motion_state_transformed()

        # scale pos command
        dpos = (
            sm_state[:3] * (self.max_pos_speed / self.frequency) * self.sm_dpos_scalar
        )

        # convert and scale rot command
        drot_xyz = sm_state[3:] * (self.max_rot_speed / self.frequency)
        drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
        drot_rotvec *= self.sm_drot_scalar
        drot = st.Rotation.from_rotvec(drot_rotvec)

        # get keyboard actions/flags
        keyboard_action, collect_enum = self.keyboard.get_action()

        # check if action is taken
        if np.allclose(dpos, 0.0) and np.allclose(drot_xyz, 0.0):
            action_taken = False
        else:
            action_taken = True

        # manage grasping
        self.steps_since_grasp += 1
        if self.steps_since_grasp < self.record_latency_when_grasping:
            action_taken = True

        self.last_grip_step += 1
        is_gripper_open = self.gripper_open
        if (
            self.sm.is_button_pressed(0)
            or self.sm.is_button_pressed(1)
            and self.last_grip_step > 10
        ):
            toggle_gripper = True

            self.gripper_open = not self.gripper_open
            self.last_grip_step = 0
            self.grasp_flag = -1 * self.grasp_flag
            self.steps_since_grasp = 0
        else:
            toggle_gripper = False

        # overwrite action from keyboard action (for screwing)
        if not (np.allclose(keyboard_action[3:6], 0.0)):
            drot = st.Rotation.from_quat(keyboard_action[3:7])
            action_taken = True

        # write out action
        new_target_pose = self.target_pose.copy()
        new_target_pose[:3] += dpos
        new_target_pose[3:] = (
            drot * st.Rotation.from_rotvec(self.target_pose[3:])
        ).as_rotvec()
        new_target_pose_mat = self.to_pose_mat(new_target_pose)

        action_struct = ActionContainer(
            next_pose_mat=new_target_pose_mat,
            grasp_flag=self.grasp_flag,
            action_taken=action_taken,
            collect_enum=collect_enum,
            is_gripper_open=is_gripper_open,
            toggle_gripper=toggle_gripper,
        )
        return action_struct

    def get_rgbd_rs(self, pipe) -> dict:
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)  # 100
            # frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            print(
                f"Couldn't get frame for device: {pipe.get_active_profile().get_device()}"
            )
            # continue
            raise

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # the .copy() here is super important!
        img_dict = dict(rgb=color_image.copy(), depth=depth_image.copy())

        return img_dict

    def get_observation(self):
        # get the rgb images
        front_rgbd = self.get_rgbd_rs(self.front_image_pipeline)
        wrist_rgbd = self.get_rgbd_rs(self.wrist_image_pipeline)

        # get the robot state
        current_ee_pose = torch.cat(self.robot.get_ee_pose(), dim=-1)
        current_joint_positions = self.robot.get_joint_positions()
        jacobian = self.robot.robot_model.compute_jacobian(current_joint_positions)
        ee_spatial_velocity = jacobian @ self.robot.get_joint_velocities()

        robot_state_dict = dict(
            ee_pos=current_ee_pose[:3].numpy(),
            ee_quat=current_ee_pose[3:7].numpy(),
            ee_pos_vel=ee_spatial_velocity[:3].numpy(),
            ee_ori_vel=ee_spatial_velocity[3:6].numpy(),
            gripper_width=self.gripper.get_state().width,
        )

        # pack these
        obs = dict(
            color_image1=wrist_rgbd["rgb"],
            color_image2=front_rgbd["rgb"],
            robot_state=robot_state_dict,
        )

        return obs

    def posemat2action(self, target_pose_mat: np.ndarray, grasp_flag: int):
        action = np.zeros(8)
        action[:3] = target_pose_mat[:-1, -1]
        action[3:7] = st.Rotation.from_matrix(target_pose_mat[:-1, :-1]).as_quat()
        action[-1] = grasp_flag

        return action


def main():
    # some main args
    frequency = args.frequency
    dt = 1 / frequency
    command_latency = args.command_latency

    # setup robot
    franka_ip = "173.16.0.1"
    # robot = RobotInterface(ip_address=franka_ip)
    robot = DiffIKWrapper(ip_address=franka_ip)

    sm_dpos_scalar = np.array([1.5] * 3)
    sm_drot_scalar = np.array([2.25] * 3)

    Kq_new = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])
    Kqd_new = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])

    robot.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new, adaptive=True)

    gripper = GripperInterface(ip_address=franka_ip)
    gripper.goto(0.08, 0.05, 0.1, blocking=False)

    # setup visuals
    zmq_url = f"tcp://127.0.0.1:{args.port_vis}"
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis["scene"].delete()

    # Setup camera streams
    rs_cfg = get_default_multi_realsense_cfg()
    resolution_width = rs_cfg.WIDTH  # pixels
    resolution_height = rs_cfg.HEIGHT  # pixels
    frame_rate = rs_cfg.FRAME_RATE  # fps

    ctx = rs.context()  # Create librealsense context for managing devices

    print(
        f"Enabling devices with serial numbers: {FRONT_CAMERA_SERIAL} (front) and {WRIST_CAMERA_SERIAL} (wrist)"
    )
    front_image_pipeline = enable_single_realsense(
        FRONT_CAMERA_SERIAL, ctx, resolution_width, resolution_height, frame_rate
    )
    time.sleep(1.0)
    wrist_image_pipeline = enable_single_realsense(
        WRIST_CAMERA_SERIAL, ctx, resolution_width, resolution_height, frame_rate
    )
    time.sleep(1.0)

    # Setup data saving
    demo_save_dir = Path(args.save_dir)
    demo_save_dir.mkdir(exist_ok=True, parents=True)
    pkl_path = demo_save_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

    episode_data = {}
    episode_data["observations"] = []
    episode_data["actions"] = []
    episode_data["furniture"] = args.furniture
    # assume all real world demos that we actually save are success
    episode_data["success"] = True

    # Setup other interfaces
    keyboard = KeyboardInterface()

    translation, quat_xyzw = robot.get_ee_pose()
    rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
    target_pose = np.array([*translation.numpy(), *rotvec])

    def polypose2target(poly_pose):
        translation, quat_xyzw = poly_pose[0], poly_pose[1]
        rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
        target_pose = np.array([*translation.numpy(), *rotvec])
        return target_pose

    def execute_gripper_action(toggle_gripper: bool, gripper_open: bool):
        if not toggle_gripper:
            return
        if gripper_open:
            gripper.grasp(0.07, 70, blocking=False)
        else:
            gripper.goto(0.08, 0.05, 0.1, blocking=False)

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
            t_start = time.monotonic()
            iter_idx = 0
            stop = False

            obs_act_helper = ObsActHelper(
                sm=sm,
                keyboard=keyboard,
                robot=robot,
                gripper=gripper,
                front_image_pipeline=front_image_pipeline,
                wrist_image_pipeline=wrist_image_pipeline,
            )

            obs_act_helper.set_target_pose(target_pose)
            obs_act_helper.set_constants(
                max_pos_speed=args.max_pos_speed,
                max_rot_speed=args.max_rot_speed,
                sm_dpos_scalar=sm_dpos_scalar,
                sm_drot_scalar=sm_drot_scalar,
                frequency=args.frequency,
            )
            print(f"Start collecting!")
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                # t_command_target = t_cycle_end + dt
                precise_wait(t_sample)

                # get robot state/image observation
                observation = obs_act_helper.get_observation()

                # get and unpack action
                action_struct = obs_act_helper.get_action()
                action_next_pose_mat = action_struct.next_pose_mat
                grasp_flag = action_struct.grasp_flag
                action_taken = action_struct.action_taken
                collect_enum = action_struct.collect_enum
                is_gripper_open = action_struct.is_gripper_open
                toggle_gripper = action_struct.toggle_gripper

                if collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                    break

                # send command to the robot
                robot.update_desired_ee_pose(action_next_pose_mat, dt=dt)
                execute_gripper_action(toggle_gripper, is_gripper_open)

                # log the data
                if action_taken:
                    action = obs_act_helper.posemat2action(
                        action_next_pose_mat, grasp_flag
                    )
                    episode_data["actions"].append(action)
                    episode_data["observations"].append(observation)

                target_pose = polypose2target(robot.get_ee_pose())
                obs_act_helper.set_target_pose(target_pose)

                # Draw the current and target pose (in meshcat)
                mc_util.meshcat_frame_show(
                    mc_vis, f"scene/target_pose", action_next_pose_mat
                )
                mc_util.meshcat_frame_show(
                    mc_vis,
                    f"scene/current_pose",
                    poly_util.polypose2mat(robot.get_ee_pose()),
                )

                precise_wait(t_cycle_end)
                iter_idx += 1

    if collect_enum == CollectEnum.SUCCESS:
        # save the data
        with open(pkl_path, "wb") as f:
            pickle.dump(episode_data, f)


if __name__ == "__main__":
    main()
