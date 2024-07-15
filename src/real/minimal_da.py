import json
from pathlib import Path
import pickle
import time
import threading
import torch
import numpy as np
import scipy.spatial.transform as st
import meshcat
import argparse
import pytorch3d.transforms as pt

from polymetis import GripperInterface, RobotInterface

from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
import threading
import torch
import numpy as np
import scipy.spatial.transform as st
import meshcat
import argparse
import pytorch3d.transforms as pt

from polymetis import GripperInterface, RobotInterface

from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from datetime import datetime

# imports for different interfaces to the cameras
from rdt.image.factory import get_realsense_rgbd_subscribers
import lcm
from rdt.image.factory import enable_realsense_devices
import pyrealsense2 as rs
from src.real.serials import WRIST_CAMERA_SERIAL, FRONT_CAMERA_SERIAL

from typing import List

from omegaconf import OmegaConf, DictConfig
from src.data_processing.utils import resize, resize_crop
from ipdb import set_trace as bp
from furniture_bench.furniture import furniture_factory
from furniture_bench.config import config
import furniture_bench.utils.transform as T

from wandb import Api
from wandb.sdk.wandb_run import Run

from src.dataset.normalizer import LinearNormalizer


api = Api()

import hydra
from omegaconf import OmegaConf

from model.diffusion.diffusion_ppo import PPODiffusion
from model.rl.gaussian_ppo import PPO_Gaussian

OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_runs(args: argparse.Namespace) -> List[Run]:
    api.flush()
    runs: List[Run] = [api.run(f"{run_id}") for run_id in args.run_id]
    return runs


def vision_encoder_field_hotfix(run, config):
    if isinstance(config.vision_encoder, str):
        # Read in the vision encoder config from the `vision_encoder` config group and set it
        OmegaConf.set_readonly(config, False)
        config.vision_encoder = OmegaConf.load(
            f"src/config/vision_encoder/{config.vision_encoder}.yaml"
        )
        OmegaConf.set_readonly(config, True)

        # Write it back to the run
        run.config = OmegaConf.to_container(config, resolve=True)
        run.update()


def quat_xyzw_to_quat_wxyz(quat):
    """Converts IsaacGym quaternion to PyTorch3D quaternion.

    IsaacGym quaternion is (x, y, z, w) while PyTorch3D quaternion is (w, x, y, z).
    """
    return torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)


def quat_wxyz_to_quat_xyzw(quat):
    """Converts IsaacGym quaternion to PyTorch3D quaternion.

    PyTorch3D quaternion is (w, x, y, z) while IsaacGym quaternion is (x, y, z, w).
    """
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)


def quat_xyzw_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    quat_wxyz = quat_xyzw_to_quat_wxyz(quat_xyzw)

    # Convert each quaternion to a rotation matrix
    rot_mats = pt.quaternion_to_matrix(quat_wxyz)

    # Extract the first two columns of each rotation matrix
    rot_6d = pt.matrix_to_rotation_6d(rot_mats)

    return rot_6d


def rot_6d_quat_xyzw(rot_6d: torch.Tensor) -> torch.Tensor:
    """Converts 6D rotation to IsaacGym quat (xyzw)."""
    rot_mat = pt.rotation_6d_to_matrix(rot_6d)
    quat_wxyz = pt.matrix_to_quaternion(rot_mat)
    quat_xyzw = quat_wxyz_to_quat_xyzw(quat_wxyz)
    return quat_xyzw


class CollectInferHelper:
    def __init__(self, args, demo_save_dir=None):
        self.args = args
        self.demo_save_dir = demo_save_dir
        self.episode_data = None

        self.init_episode()

    def init_episode(self):
        print(f"New episode")
        episode_data = {}
        episode_data["observations"] = []
        episode_data["actions"] = []
        episode_data["furniture"] = self.args.furniture
        # assume all real world demos that we actually save are success
        episode_data["success"] = True
        episode_data["args"] = self.args.__dict__

        self.episode_data = episode_data
        self.is_recording = False
        return episode_data

    def set_recording(self, record: bool):
        if self.is_recording:
            return
        self.is_recording = record
        if record:
            print(f"Starting to record...")

    @staticmethod
    def to_isaac_dpose_from_abs(obs, abs_action, rm=True):
        """
        Convert from absolute current and desired pose to delta pose

        Args:
            rm (bool): 'rm' stands for 'right multiplication' - If True, assume commands send as right multiply (local rotations)
        """
        # get the current pose mat from the observation
        current_pose_mat = np.eye(4)
        current_pose_mat[:-1, -1] = obs["robot_state"][0, :3].cpu().numpy()
        current_pose_mat[:-1, :-1] = st.Rotation.from_quat(
            obs["robot_state"][0, 3:7].cpu().numpy()
        ).as_matrix()

        # get the absolute goal pose from the action
        goal_pose_mat = np.eye(4)
        goal_pose_mat[:-1, -1] = abs_action[0, :3].cpu().numpy()
        goal_pose_mat[:-1, :-1] = st.Rotation.from_quat(
            abs_action[0, 3:7].cpu().numpy()
        ).as_matrix()

        # get the grasp flag
        grasp_flag = abs_action[-1].cpu().numpy().reshape(-1)

        # convert to delta
        if rm:
            delta_rot_mat = (
                np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
            )
        else:
            delta_rot_mat = goal_pose_mat[:-1:-1] @ np.linalg.inv(
                current_pose_mat[:-1, :-1]
            )

        target_translation = goal_pose_mat[:-1, -1] - current_pose_mat[:-1, -1]
        target_quat_xyzw = st.Rotation.from_matrix(delta_rot_mat).as_quat()

        target_dpose = np.concatenate(
            (target_translation, target_quat_xyzw, grasp_flag), axis=-1
        ).reshape(1, -1)

        return target_dpose

    def log(self, abs_action, obs):
        delta_action = self.to_isaac_dpose_from_abs(obs=obs, abs_action=abs_action)
        self.episode_data["actions"].append(delta_action)
        self.episode_data["observations"].append(obs)

    def save_pkl(self, init_new=True):
        if not self.is_recording:
            print(
                f'Cannot save pkl unless "is_recording" flag is set with "set_is_recording"!'
            )
            return

        # save the data
        pkl_path = (
            self.demo_save_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"
        )
        print(
            f"Saving trajectory with {len(self.episode_data['actions'])} transitions to folder: {pkl_path}"
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(self.episode_data, f)

        if init_new:
            self.init_episode()


import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-p", "--port_vis", type=int, default=6000)
# parser.add_argument("--frequency", type=int, default=10)  # 30
# parser.add_argument("--command_latency", type=float, default=0.01)
# parser.add_argument("--deadzone", type=float, default=0.05)
# parser.add_argument("--max_pos_speed", type=float, default=0.3)
# parser.add_argument("--max_rot_speed", type=float, default=0.7)
# parser.add_argument("--save_dir", required=alse)
# parser.add_argument("--use_lcm", action="store_true")

# parser.add_argument("--run-id", type=str, required=False)
# parser.add_argument("--gpu", type=int, default=0)
# parser.add_argument("--wandb", action="store_true")
# parser.add_argument("--leaderboard", action="store_true")
# parser.add_argument("--project-id", type=str, default=None)
# parser.add_argument(
#     "--action-type", type=str, default="delta", choices=["delta", "pos"]
# )
# parser.add_argument("--verbose", "-v", action="store_true")
# parser.add_argument("--multitask", action="store_true")
# parser.add_argument("-ex", "--execute", action="store_true")
# parser.add_argument("-o", "--observation-type", type=str, default="image")
# parser.add_argument("-w", "--wts-name", default="best", type=str)
# parser.add_argument("--log-ci", action="store_true")
# parser.add_argument("--ci-save-dir", type=str, default=None)
# parser.add_argument("--furniture", type=str, default=None)

# args = parser.parse_args()

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


# NOTE: Should we allow this to silently happen?
def resize_image(obs, key):
    try:
        obs[key] = resize(obs[key])
    except KeyError:
        pass


def resize_crop_image(obs, key):
    try:
        obs[key] = resize_crop(obs[key])
    except KeyError:
        pass


def convert_reference_frame_mat_tedrake(
    pose_source_mat, pose_frame_target_mat, pose_frame_source_mat
):

    # transform that maps from target to source (S = XT)
    target2source_mat = np.linalg.inv(pose_frame_target_mat) @ pose_frame_source_mat

    # obtain source pose in target frame
    pose_source_in_target_mat = target2source_mat @ pose_source_mat
    return pose_source_in_target_mat


def convert_reference_frame_mat(
    pose_source_mat, pose_frame_target_mat, pose_frame_source_mat
):

    # transform that maps from target to source (S = XT)
    target2source_mat = np.matmul(
        pose_frame_source_mat, np.linalg.inv(pose_frame_target_mat)
    )

    # obtain source pose in target frame
    pose_source_in_target_mat = np.matmul(target2source_mat, pose_source_mat)
    return pose_source_in_target_mat


def convert_tip2wrist(tip_pose_mat):

    tip2wrist_tf_mat = np.eye(4)
    tip2wrist_tf_mat[:-1, -1] = np.array([0.0, 0.0, -0.1034])
    tip2wrist_tf_mat[:-1, :-1] = st.Rotation.from_quat(
        [0.0, 0.0, 0.3826834323650898, 0.9238795325112867]
    ).as_matrix()

    wrist_pose_mat = convert_reference_frame_mat(
        pose_source_mat=tip2wrist_tf_mat,
        pose_frame_target_mat=np.eye(4),
        pose_frame_source_mat=tip_pose_mat,
    )

    return wrist_pose_mat


def convert_wrist2tip(wrist_pose_mat):

    wrist2tip_tf_mat = np.eye(4)
    wrist2tip_tf_mat[:-1, -1] = np.array([0.0, 0.0, 0.1034])
    wrist2tip_tf_mat[:-1, :-1] = st.Rotation.from_quat(
        [0.0, 0.0, -0.3826834323650898, 0.9238795325112867]
    ).as_matrix()

    tip_pose_mat = convert_reference_frame_mat(
        pose_source_mat=wrist2tip_tf_mat,
        pose_frame_target_mat=np.eye(4),
        pose_frame_source_mat=wrist_pose_mat,
    )

    return tip_pose_mat


class SimpleDiffIKFrankaEnv:
    def __init__(
        self,
        mc_vis: meshcat.Visualizer,
        robot: RobotInterface,
        gripper: GripperInterface,
        use_lcm: bool = False,
        device: str = "cuda",
        execute: bool = False,
        observation_type: str = "image",
        proprioceptive_state_dim: int = 10,
        control_mode: str = "pos",
        action_horizon: int = 10,
        furniture_name: str = "one_leg",
        part_reset_poses: torch.Tensor = None,
    ):

        self.device = device
        self.execute = execute

        self.mc_vis = mc_vis
        self.robot = robot
        self.gripper = gripper
        self.gripper_open = True
        self.last_grasp = torch.Tensor([-1]).float().to(self.device).squeeze()
        self.last_grip_step = 0

        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open and close actions

        # flags
        self.observation_type = observation_type
        self.proprioceptive_state_dim = proprioceptive_state_dim
        self.control_mode = control_mode

        self.robot_home = torch.tensor(
            [-0.0046, 0.2833, 0.0664, -2.2494, -0.1338, 2.5323, 0.0317]
        )

        self.Kq_new = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])
        self.Kqd_new = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])

        # Move the robot to home to begin with
        self.gripper.goto(0.08, 0.05, 0.1, blocking=False)
        self.robot.move_to_joint_positions(self.robot_home)

        if self.observation_type == "image":
            self.get_obs = self.get_img_obs
            if use_lcm:
                self._setup_lcm()
                self.get_rgbd = self.get_rgbd_lcm
            else:
                self._setup_rs2()
                self.get_rgbd = self.get_rgbd_rs
        else:
            self.get_obs = self.get_state_obs
            self.furniture = furniture_factory(furniture_name)
            print(f"Starting detection")
            self.furniture.start_detection()

            self.part_reset_poses = part_reset_poses

        self.action_horizon = action_horizon
        self.act_step = 0

        self.cached_leg_pose_mat_ee = None
        self.last_parts_poses = None

    def set_timing(
        self, iter_idx: int = 0, dt: float = 0.1, command_latency: float = 0.01
    ):
        self.iter_idx = iter_idx
        self.dt = dt
        self.command_latency = command_latency

    def set_start(self, t_start):
        self.t_start = t_start

    def reset_timing(self):
        self.iter_idx = 0

    def _setup_lcm(self):

        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        rs_cfg = get_default_multi_realsense_cfg()
        self.image_subs = get_realsense_rgbd_subscribers(lc, rs_cfg)

        def lc_th(lc):
            while True:
                lc.handle_timeout(1)
                time.sleep(0.001)

        lc_thread = threading.Thread(target=lc_th, args=(lc,))
        lc_thread.daemon = True
        lc_thread.start()

    def _setup_rs2(self):
        rs_cfg = get_default_multi_realsense_cfg()
        resolution_width = rs_cfg.WIDTH  # pixels
        resolution_height = rs_cfg.HEIGHT  # pixels
        frame_rate = rs_cfg.FRAME_RATE  # fps

        ctx = rs.context()  # Create librealsense context for managing devices
        # serials = rs_cfg.SERIAL_NUMBERS
        serials = [WRIST_CAMERA_SERIAL, FRONT_CAMERA_SERIAL]

        print(f"Enabling devices with serial numbers: {serials}")
        self.image_pipelines = enable_realsense_devices(
            serials, ctx, resolution_width, resolution_height, frame_rate
        )

    def get_rgbd_lcm(self):
        rgbd_list = []
        for name, img_sub, info_sub in self.image_subs:
            rgb_image, depth_image = img_sub.get_rgb_and_depth(block=True)
            if rgb_image is None or depth_image is None:
                return

            img_dict = dict(rgb=rgb_image, depth=depth_image)
            rgbd_list.append(img_dict)

        return rgbd_list

    def get_rgbd_rs(self):
        rgbd_list = []

        align_to = rs.stream.color
        align = rs.align(align_to)

        for device, pipe in self.image_pipelines:
            try:
                # Get frameset of color and depth
                frames = pipe.wait_for_frames(100)
            except RuntimeError as e:
                print(f"Couldn't get frame for device: {device}")
                # continue
                raise

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            img_dict = dict(rgb=color_image.copy(), depth=depth_image.copy())
            rgbd_list.append(img_dict)

        return rgbd_list

    def action2pose_mat(self, raw_action: torch.Tensor):
        # input action is pos and 6d rot
        rot_6d = raw_action[3:9]
        rot_mat = pt.rotation_6d_to_matrix(rot_6d)
        action_pose_mat = torch.eye(4)
        action_pose_mat[:-1, :-1] = rot_mat
        action_pose_mat[:-1, -1] = raw_action[:3]

        return action_pose_mat

    def get_robot_state(self):
        # convert to tip
        current_ee_wrist_pose_mat = poly_util.polypose2mat(self.robot.get_ee_pose())
        current_ee_tip_pose_mat = convert_wrist2tip(current_ee_wrist_pose_mat)
        pos, quat_xyzw = poly_util.mat2polypose(current_ee_tip_pose_mat)

        # rot_6d = quat_xyzw_to_rot_6d(quat_xyzw)
        # robot_state = torch.cat([pos, rot_6d], dim=-1)
        robot_state = torch.cat([pos, quat_xyzw], dim=-1).float()

        # If current policy also expects EE velocities
        if self.proprioceptive_state_dim == 16:
            current_joint_pos = self.robot.get_joint_positions()
            jacobian = self.robot.robot_model.compute_jacobian(current_joint_pos)
            ee_vel = jacobian @ self.robot.get_joint_velocities()

            robot_state = torch.cat([robot_state, ee_vel], dim=-1)

        gripper_width = torch.Tensor([self.gripper.get_state().width])
        robot_state = torch.cat([robot_state, gripper_width], dim=-1)

        # convert to tensors
        robot_state = robot_state.reshape(1, -1).to(self.device)
        return robot_state

    def get_state_obs(self):

        robot_state = self.get_robot_state()

        parts_poses = self.part_reset_poses.clone()

        detection_sleep = 0.1
        distance_strikes = 0
        n_times_slept = 0

        while True:
            parts_poses_april = (
                torch.tensor(self.furniture.get_parts_poses()[0])
                .reshape(1, -1)
                .to(self.device)
            )

            if torch.allclose(
                parts_poses_april[0, :7], torch.tensor([0.0], device=self.device)
            ):
                print("sleeping (top)")
                time.sleep(detection_sleep)
                continue
                # print(f"Replacing top with cached")
                # parts_poses_april[0, :7] = self.last_parts_poses[0, :7]

            # Distance leg moved since last time
            # if self.last_parts_poses is not None and (
            #     torch.norm(parts_poses[0, 28:31] - self.last_parts_poses[0, 28:31]) > 0.015
            #     or torch.norm(parts_poses[0, :3] - self.last_parts_poses[0, :3]) > 0.015
            # ):

            moved_dist = (
                torch.norm(
                    parts_poses_april[0, 28:31] - self.last_parts_poses[0, 28:31]
                )
                if self.last_parts_poses is not None
                else 0
            )

            # if moved_dist > 0.15:
            #     print(f"Moved: {moved_dist}")
            #     continue

            if self.cached_leg_pose_mat_ee is not None:
                # compute april frame leg pose based on cached in hand pose
                print("Using hand cached leg pose")
                leg_pose_ee = self.cached_leg_pose_mat_ee
                ee_pose_mat_world = convert_wrist2tip(
                    poly_util.polypose2mat(self.robot.get_ee_pose())
                )

                leg_pose_april_mat = convert_reference_frame_mat_tedrake(
                    pose_source_mat=leg_pose_ee,
                    pose_frame_target_mat=config["robot"]["tag_base_from_robot_base"],
                    pose_frame_source_mat=ee_pose_mat_world,
                )
                parts_poses_april[0, 28:35] = (
                    torch.from_numpy(np.concatenate(T.mat2pose(leg_pose_april_mat)))
                    .float()
                    .to(self.device)
                )

            if (
                torch.allclose(
                    parts_poses_april[0, 28:35], torch.tensor([0.0], device=self.device)
                )
                # or moved_cm > 0.02
            ):

                # if moved_cm > 0.02:
                #     # Update the last leg pose
                #     distance_strikes += 1
                #     print(f"Distance strikes: {distance_strikes}")
                #     if distance_strikes > 1:
                #         self.last_parts_poses[0, 28:35] = parts_poses_april[
                #             0, 28:35
                #         ].clone()
                #         distance_strikes = 0

                if self.cached_leg_pose_mat_ee is not None:
                    # compute april frame leg pose based on cached in hand pose
                    print("Using hand cached leg pose")
                    leg_pose_ee = self.cached_leg_pose_mat_ee
                    ee_pose_mat_world = convert_wrist2tip(
                        poly_util.polypose2mat(self.robot.get_ee_pose())
                    )

                    leg_pose_april_mat = convert_reference_frame_mat_tedrake(
                        pose_source_mat=leg_pose_ee,
                        pose_frame_target_mat=config["robot"][
                            "tag_base_from_robot_base"
                        ],
                        pose_frame_source_mat=ee_pose_mat_world,
                    )
                    parts_poses_april[0, 28:35] = (
                        torch.from_numpy(np.concatenate(T.mat2pose(leg_pose_april_mat)))
                        .float()
                        .to(self.device)
                    )
                else:
                    if n_times_slept < 5:
                        print("sleeping (leg)")
                        n_times_slept += 1
                        time.sleep(detection_sleep)
                        continue
                    else:
                        print(f"Replacing leg with cached")
                        n_times_slept = 0
                        parts_poses_april[0, 28:35] = self.last_parts_poses[0, 28:35]

            break

        parts_poses[0, :7] = parts_poses_april[0, :7]
        parts_poses[0, 28:35] = parts_poses_april[0, 28:35]

        self.last_parts_poses = parts_poses.clone()

        top_pos = slice(0, 3)
        leg4_pos = slice(28, 31)

        parts_poses[:, leg4_pos] += torch.tensor([0.0, 0.0, 0.0028], device=self.device)
        parts_poses[:, top_pos] += torch.tensor([0.0, 0.0, 0.0028], device=self.device)

        leg_pose_mat_tag = T.pose2mat(parts_poses.squeeze()[28:35].cpu().numpy())
        leg_pose_mat = convert_reference_frame_mat(
            pose_source_mat=leg_pose_mat_tag,
            pose_frame_target_mat=np.eye(4),
            pose_frame_source_mat=config["robot"]["tag_base_from_robot_base"],
        )
        self.last_leg_pose_mat_world = leg_pose_mat

        if self.mc_vis is not None:

            # parts_poses = torch.cat((parts_poses, self.obstacle_pose), dim=-1)

            # visualize poses
            leg_reset_pose_mat_tag = T.pose2mat(
                self.part_reset_poses.squeeze()[28:35].cpu().numpy()
            )
            leg_reset_pose_mat = convert_reference_frame_mat(
                pose_source_mat=leg_reset_pose_mat_tag,
                pose_frame_target_mat=np.eye(4),
                pose_frame_source_mat=config["robot"]["tag_base_from_robot_base"],
            )

            mc_util.meshcat_frame_show(
                self.mc_vis, f"scene/leg_reset_pose", leg_reset_pose_mat
            )

            mc_util.meshcat_frame_show(self.mc_vis, f"scene/leg_pose", leg_pose_mat)

            top_reset_pose_mat_tag = T.pose2mat(
                self.part_reset_poses.squeeze()[0:7].cpu().numpy()
            )
            top_reset_pose_mat = convert_reference_frame_mat(
                pose_source_mat=top_reset_pose_mat_tag,
                pose_frame_target_mat=np.eye(4),
                pose_frame_source_mat=config["robot"]["tag_base_from_robot_base"],
            )
            mc_util.meshcat_frame_show(
                self.mc_vis, f"scene/top_reset_pose", top_reset_pose_mat
            )

            top_pose_mat_tag = T.pose2mat(parts_poses.squeeze()[0:7].cpu().numpy())
            top_pose_mat = convert_reference_frame_mat(
                pose_source_mat=top_pose_mat_tag,
                pose_frame_target_mat=np.eye(4),
                pose_frame_source_mat=config["robot"]["tag_base_from_robot_base"],
            )
            mc_util.meshcat_frame_show(self.mc_vis, f"scene/top_pose", top_pose_mat)

        obs = dict(
            parts_poses=torch.clamp(parts_poses, -1.5, 1.5),
            robot_state=torch.clamp(robot_state, -1.5, 1.5),
        )

        return obs

    def get_img_obs(self):
        # get observations
        # NOTE: In the simulator we get a 14-dimensional vector with proprioception:
        # [pos, quat, vel, ang vel, gripper], while here we only get 7(?)
        robot_state = self.get_robot_state()

        rgbd_list = self.get_rgbd()
        if rgbd_list is None:
            raise ValueError("Could not get list of RGB-D images!")
        # Quaternions follow the convention of <x, y, z, w>
        # pos, quat_xyzw = self.robot.get_ee_pose()

        img1_tensor = torch.from_numpy(rgbd_list[0]["rgb"]).unsqueeze(0).to(self.device)
        img2_tensor = torch.from_numpy(rgbd_list[1]["rgb"]).unsqueeze(0).to(self.device)

        obs = dict(
            color_image1=img1_tensor,
            color_image2=img2_tensor,
            robot_state=robot_state,
        )
        return obs

    def reset(self):

        self.gripper.goto(0.08, 0.05, 0.1, blocking=False)
        self.robot.move_to_joint_positions(self.robot_home)
        self.robot.start_joint_impedance(
            Kq=self.Kq_new, Kqd=self.Kqd_new, adaptive=True
        )

        self.cached_leg_pose_mat_ee = None
        self.last_parts_poses = None

        self.last_grip_step = 0
        self.last_grasp = torch.Tensor([-1]).float().to(self.device).squeeze()

        self.gripper_open = True

        self.reset_timing()

        return self.get_obs()

    def step(self, action: torch.Tensor):

        # calculate timing
        t_cycle_end = self.t_start + (self.iter_idx + 1) * self.dt
        t_sample = t_cycle_end - self.command_latency
        # t_command_target = t_cycle_end + dt
        precise_wait(t_sample)

        # convert action
        new_target_pose_mat = self.action2pose_mat(action)

        # If the action type is delta, we need to apply it to the
        # current state so we get the desired pose
        if self.control_mode == "delta":
            ee_pose_mat = torch.from_numpy(
                poly_util.polypose2mat(self.robot.get_ee_pose())
            ).to(torch.float32)
            new_target_pose_mat = torch.matmul(new_target_pose_mat, ee_pose_mat)

        self.last_grip_step += 1
        grasp = action[-1]
        grasp_cond1 = (torch.sign(grasp) != torch.sign(self.last_grasp)).item()
        grasp_cond2 = (torch.abs(grasp) > self.grasp_margin).item()
        grasp_cond3 = self.last_grip_step > 10

        if grasp_cond1 and grasp_cond2 and grasp_cond3:
            # if grasp != self.last_grasp and last_grip_step > 10:
            # gripper.gripper_close() if gripper_open else gripper.gripper_open()
            if self.gripper_open:
                self.gripper.grasp(0.07, 70, blocking=False)

                # We need to wait until the object is fully grasped before we can
                # check what we grasped and store the pose of the leg if that was the object
                while (
                    not self.gripper.get_state().is_grasped
                ) or self.gripper.get_state().is_moving:
                    time.sleep(0.01)

                # Grasp width leg:      0.02859 meters
                # Grasp width tabletop: 0.00569 meters
                # Grasp width nothing:  0.0     meters

                # Now the gripper is closed, if the grasp width is more than 0.02, we assume
                # we have grasped the leg, and only then do we store the pose of the leg in the hand
                # This fixes 2 issues: (1) The cached pose of the leg could be offset from the true,
                # and (2) we could tie the leg to the gripper when we grasp the tabletop

                if self.gripper.get_state().width > 0.02:
                    # get the current EE pose and leg pose
                    ee_pose_mat_world = convert_wrist2tip(
                        poly_util.polypose2mat(self.robot.get_ee_pose())
                    )
                    leg_pose_mat_world = self.last_leg_pose_mat_world
                    # cache the in-hand pose for use when april tag detection is bad
                    self.cached_leg_pose_mat_ee = convert_reference_frame_mat(
                        pose_source_mat=leg_pose_mat_world,
                        pose_frame_target_mat=ee_pose_mat_world,
                        pose_frame_source_mat=np.eye(4),
                    )
            else:
                # goto for opening
                self.gripper.goto(0.08, 0.05, 0.1, blocking=False)

                # Blank the cached pose of the leg in the hand
                self.cached_leg_pose_mat_ee = None

                # Just sleep for a hot minute to let the gripper open a bit
                time.sleep(0.1)

            self.gripper_open = not self.gripper_open
            self.last_grip_step = 0
            self.last_grasp = grasp

        # clip robot z coordinate
        new_target_pose_mat[2, -1] = np.clip(
            new_target_pose_mat[2, -1], a_min=0.02, a_max=None
        )

        if self.execute:
            # self.robot.update_desired_ee_pose(
            #     new_target_pose_mat, dt=self.dt
            # )  # , scalar=0.5)
            self.robot.update_desired_ee_pose(
                convert_tip2wrist(new_target_pose_mat), dt=self.dt
            )

        if self.mc_vis is not None:
            # # Draw the current target pose (in meshcat)
            # mc_util.meshcat_frame_show(
            #     self.mc_vis,
            #     f"scene/target_pose_wrist",
            #     convert_tip2wrist(new_target_pose_mat.cpu().numpy()),
            # )
            mc_util.meshcat_frame_show(
                self.mc_vis, f"scene/target_pose_tip", new_target_pose_mat.cpu().numpy()
            )
            mc_util.meshcat_frame_show(
                self.mc_vis,
                f"scene/current_pose_tip",
                convert_wrist2tip(poly_util.polypose2mat(self.robot.get_ee_pose())),
            )
            # mc_util.meshcat_frame_show(
            #     self.mc_vis,
            #     f"scene/current_pose_wrist",
            #     poly_util.polypose2mat(self.robot.get_ee_pose()),
            # )

        # calculate timing
        precise_wait(t_cycle_end)
        self.iter_idx += 1

        # self.act_step += 1
        # if self.act_step < self.action_horizon:
        #     # time.sleep(0.01)
        #     time.sleep(0.025)
        # else:
        #     self.act_step = 0

        time.sleep(0.025)

        # get new obs
        new_obs = self.get_obs()

        return new_obs, None, None, None


def proprioceptive_quat_to_6d_rotation(robot_state: torch.tensor) -> torch.tensor:
    """
    Convert the 14D proprioceptive state space to 16D state space.

    Parts:
        - 3D position
        - 4D quaternion rotation
        - 3D linear velocity
        - 3D angular velocity
        - 1D gripper width

    Rotation 4D quaternion -> 6D vector represention

    Accepts any number of leading dimensions.
    """
    # assert robot_state.shape[-1] == 14, "Robot state must be 14D"

    # Get each part of the robot state
    pos = robot_state[..., :3]  # (x, y, z)
    ori_quat = robot_state[..., 3:7]  # (x, y, z, w)
    pos_vel = robot_state[..., 7:10]  # (x, y, z)
    ori_vel = robot_state[..., 10:13]  # (x, y, z)
    gripper = robot_state[..., 13:]  # (width)

    # Convert quaternion to 6D rotation
    ori_6d = isaac_quat_to_rot_6d(ori_quat)

    # Concatenate all parts
    robot_state_6d = torch.cat([pos, ori_6d, pos_vel, ori_vel, gripper], dim=-1)

    return robot_state_6d


def isaac_quat_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    # quat_wxyz = isaac_quat_to_pytorch3d_quat(quat_xyzw)

    # Convert each quaternion to a rotation matrix
    rot_mats = quaternion_to_matrix(quat_xyzw)

    # Extract the first two columns of each rotation matrix
    rot_6d = matrix_to_rotation_6d(rot_mats)

    return rot_6d


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


class FurnitureRLSimEnvMultiStepWrapper:

    def __init__(
        self,
        env: SimpleDiffIKFrankaEnv,
        n_action_steps=8,
        normalization_path=None,
        device="cuda",
    ):

        self.n_action_steps = n_action_steps
        self.env = env
        self.device = device

        # set up normalization
        self.normalize = normalization_path is not None
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(
            torch.load(normalization_path, map_location=device)
        )

    def reset(
        self,
        **kwargs,
    ):
        """Resets the environment."""
        obs = self.env.reset()
        nobs = self.process_obs(obs)

        return nobs

    def reset_arg(self, options_list=None):
        return self.reset()

    def reset_one_arg(self, env_ind=None, options=None):
        if env_ind is not None:
            env_ind = torch.tensor([env_ind], device=self.device)

        return self.reset()

    def step(self, action: torch.Tensor):
        """
        Takes in a chunk of actions of length n_action_steps
        and steps the environment n_action_steps times
        and returns an aggregated observation, reward, and done signal
        """
        # action: (n_envs, n_action_steps, action_dim)

        # Denormalize the action
        action = self.normalizer(action, "actions", forward=False)

        # Step the environment n_action_steps times
        obs, reward, done, info = self._inner_step(action)

        nobs: torch.Tensor = self.process_obs(obs)

        return nobs, reward, done, info

    def _inner_step(self, action_chunk: torch.Tensor):
        for i in range(self.n_action_steps):
            # The dimensions of the action_chunk are (num_envs, chunk_size, action_dim)
            obs, reward, done, info = self.env.step(action_chunk[0, i, :])
            # time.sleep(0.001)

        return obs, reward, done, info

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        robot_state = obs["robot_state"]

        # Convert the robot state to have 6D pose
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        parts_poses = obs["parts_poses"]

        obs = torch.cat([robot_state, parts_poses], dim=-1)
        nobs = self.normalizer(obs, "observations", forward=True)
        nobs = torch.clamp(nobs, -5, 5)

        # Insert a dummy dimension for the n_obs_steps (n_envs, obs_dim) -> (n_envs, n_obs_steps, obs_dim)
        nobs = nobs.unsqueeze(1)  # .cpu().numpy()

        return nobs


run_idx = 0  # np.random.randint(0, 3)

run_list = [
    # DPPO after 200 iterations
    {
        "wt_path": Path(
            "/home/anthony/repos/research/robust-rearrangement/models/one_leg_low/one_leg_low_dim_ft_diffusion_unet_ta16_td100_tdf-5/2024-07-04_17-04-28/checkpoint/state_200.pt"
        ),
        "cfg_path": Path(
            "/home/anthony/repos/research/robust-rearrangement/models/one_leg_low/one_leg_low_dim_ft_diffusion_unet_ta16_td100_tdf-5/2024-07-04_17-04-28/.hydra"
        ),
    },
    # DP after BC pretraining
    {
        "wt_path": Path(
            "/home/anthony/repos/research/robust-rearrangement/models/one_leg_low/one_leg_low_dim_ft_diffusion_unet_ta16_td100_tdf-5/2024-07-04_17-04-28/checkpoint/state_0.pt"
        ),
        "cfg_path": Path(
            "/home/anthony/repos/research/robust-rearrangement/models/one_leg_low/one_leg_low_dim_ft_diffusion_unet_ta16_td100_tdf-5/2024-07-04_17-04-28/.hydra"
        ),
    },
    # Gaussian MLP after 200 iterations
    {
        "wt_path": Path(
            "/home/anthony/repos/research/robust-rearrangement/models/one_leg_low/one_leg_low_dim_ft_gaussian_mlp_ta8/2024-07-02_11-57-36/checkpoint/state_200.pt"
        ),
        "cfg_path": Path(
            "/home/anthony/repos/research/robust-rearrangement/models/one_leg_low/one_leg_low_dim_ft_gaussian_mlp_ta8/2024-07-02_11-57-36/.hydra"
        ),
    },
]


normalization_path = "/home/anthony/repos/research/robust-rearrangement/models/one_leg_low/normalization.pth"


@hydra.main(
    config_path=str(run_list[run_idx]["cfg_path"]),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    # Make the device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    OmegaConf.resolve(cfg)
    cfg.model.network_path = str(run_list[run_idx]["wt_path"])

    actor: PPODiffusion | PPO_Gaussian = hydra.utils.instantiate(
        cfg.model,
    )

    frequency = 10
    dt = 1 / frequency
    command_latency = 0.01

    franka_ip = "173.16.0.1"
    # robot = RobotInterface(ip_address=franka_ip)
    robot = DiffIKWrapper(ip_address=franka_ip)
    gripper = GripperInterface(ip_address=franka_ip)

    zmq_url = f"tcp://127.0.0.1:6001"
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis["scene"].delete()

    # Load the trajectory from a pickle
    import zarr

    zarr_path = "/home/anthony/repos/research/robust-rearrangement/src/real/sample_sim_trajectories/success.zarr"
    z = zarr.open(zarr_path, mode="r")

    ep_end = z["episode_ends"][0]
    robot_state = torch.from_numpy(z["robot_state"][:ep_end, :])
    parts_poses = torch.from_numpy(z["parts_poses"][:ep_end, :])

    # Initial parts poses from the demo dataset
    demo_init_parts_poses = parts_poses[:1].to(device)

    # simple env wrapper
    env = SimpleDiffIKFrankaEnv(
        mc_vis=mc_vis,
        robot=robot,
        gripper=gripper,
        use_lcm=False,
        execute=cfg.get("execute", False),
        observation_type="state",
        proprioceptive_state_dim=16,
        control_mode="pos",
        furniture_name="one_leg",
        part_reset_poses=demo_init_parts_poses,
    )

    # Setup data saving
    # eval_save_dir = Path("test_eval_dppo") / str(datetime.now())
    # eval_save_dir.mkdir(exist_ok=True, parents=True)

    episode_dict = {}
    episode_dict["robot_state"] = []
    episode_dict["image_front"] = []
    episode_dict["image_wrist"] = []
    episode_dict["actions"] = []

    actor.eval()
    actor.cuda()

    # mc
    # actor.set_mc(mc_vis)

    env = FurnitureRLSimEnvMultiStepWrapper(
        env=env,
        normalization_path=normalization_path,
    )

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    keyboard = KeyboardInterface()
    f = open(f"outputs/eval_run_{run_idx=}_{start_time=}.csv", mode="w")

    f.write(
        f"Starting evaluation for run {run_idx} at {start_time}\n"
        f"Using model: {run_list[run_idx]['wt_path']}\n"
        f"Using config: {run_list[run_idx]['cfg_path']}\n"
    )

    i = 0
    while True:

        obs = env.reset()
        i += 1

        t_start = time.monotonic()

        env.env.set_start(t_start)
        env.env.set_timing(iter_idx=0, dt=dt, command_latency=command_latency)

        print(f"Starting evaluation...\n\n\n")
        time.sleep(1.0)
        running = True

        alpha = 0.0
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        f.write(f"Starting episode {i} at {start_time}\n")

        leg4_pos = slice(16 + 28, 16 + 31)
        top_pos = slice(16 + 0, 16 + 3)

        while True:  # and i < len(actions):
            # Catch keyboard press, and potentially start recording new episode
            _, collect_enum = keyboard.get_action()

            if collect_enum == CollectEnum.PAUSE_HOLD:
                running = not running

            if collect_enum == CollectEnum.RESET:
                print(f"Activating pause and breaking out of loop")
                running = False
                break

            if not running:
                print(f"Pause button pressed")
                time.sleep(1.0)
                continue

            # Concatenate the obs to the running obs

            # Print the current std for each dim of the running obs

            samples = actor(
                # cond=torch.from_numpy(obs).float().to(device),
                cond=obs,
                deterministic=True,
                return_chain=True,
            )
            output_venv = samples.trajectories  # n_env x horizon x act
            # output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
            action_venv = output_venv[:, :8]

            old_parts_poses = obs.clone()

            obs, reward, done, _ = env.step(action_venv)

            obs[:, leg4_pos] = (
                alpha * old_parts_poses[:, leg4_pos] + (1 - alpha) * obs[:, leg4_pos]
            )

            obs[:, top_pos] = (
                alpha * old_parts_poses[:, top_pos] + (1 - alpha) * obs[:, top_pos]
            )


if __name__ == "__main__":
    main()
