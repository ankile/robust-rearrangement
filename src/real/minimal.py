import os, os.path as osp
from pathlib import Path
import sys
import pickle
import collections
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
from src.behavior.base import Actor  # noqa
from src.behavior import get_actor
from src.data_processing.utils import resize, resize_crop
from ipdb import set_trace as bp
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action
from furniture_bench.furniture import furniture_factory

from wandb import Api
from wandb.sdk.wandb_run import Run

api = Api()


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

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port_vis", type=int, default=6000)
parser.add_argument("--frequency", type=int, default=10)  # 30
parser.add_argument("--command_latency", type=float, default=0.01)
parser.add_argument("--deadzone", type=float, default=0.05)
parser.add_argument("--max_pos_speed", type=float, default=0.3)
parser.add_argument("--max_rot_speed", type=float, default=0.7)
parser.add_argument("--save_dir", required=True)
parser.add_argument("--use_lcm", action="store_true")

parser.add_argument("--run-id", type=str, required=False)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--leaderboard", action="store_true")
parser.add_argument("--project-id", type=str, default=None)
parser.add_argument(
    "--action-type", type=str, default="pos", choices=["delta", "pos"]
)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("-ex", "--execute", action="store_true")
parser.add_argument("-o", "--observation-type", type=str, default="image")
parser.add_argument("-w", "--wts-name", default="best", type=str)
parser.add_argument("--log-ci", action="store_true")
parser.add_argument("--ci-save-dir", type=str, default=None)
parser.add_argument("--furniture", type=str, default=None)

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
        part_poses_norm_mid: torch.Tensor = None,
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
            self.obstacle_pose = torch.tensor(
                [[0.0069, 0.3629, -0.0150, -1.0000, 0.0000, 0.0000, 0.0000]],
                device=self.device,
            )

            part_reset_poses = []
            for i in range(len(self.furniture.parts)):
                reset_pos = self.furniture.parts[i].reset_pos[0].tolist()
                reset_quat = (
                    st.Rotation.from_matrix(
                        self.furniture.parts[i].reset_ori[0][:-1, :-1]
                    )
                    .as_quat()
                    .tolist()
                )
                part_reset_poses.append(reset_pos + reset_quat)
            # self.part_reset_poses = (
            #     torch.tensor(part_reset_poses).to(self.device).reshape(-1)
            # )
            self.part_reset_poses = part_poses_norm_mid.to(self.device)[:35]

        self.action_horizon = action_horizon
        self.act_step = 0

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

        parts_poses = (
            torch.tensor(self.furniture.get_parts_poses()[0])
            .reshape(1, -1)
            .to(self.device)
        )
        parts_poses[0, 7:28] = self.part_reset_poses[7:28]
        # parts_poses[0, :] = self.part_reset_poses[:]
        parts_poses = torch.cat((parts_poses, self.obstacle_pose), dim=-1)

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

    def step(self, action: torch.Tensor):

        # # calculate timing
        # t_cycle_end = self.t_start + (self.iter_idx + 1) * self.dt
        # t_sample = t_cycle_end - self.command_latency
        # # t_command_target = t_cycle_end + dt
        # precise_wait(t_sample)

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
        grasp_cond3 = self.last_grip_step > 20

        if grasp_cond1 and grasp_cond2 and grasp_cond3:
            # if grasp != self.last_grasp and last_grip_step > 10:
            # gripper.gripper_close() if gripper_open else gripper.gripper_open()
            if self.gripper_open:
                self.gripper.grasp(0.07, 70, blocking=False)
            else:
                # goto for opening
                self.gripper.goto(0.08, 0.05, 0.1, blocking=False)

            self.gripper_open = not self.gripper_open
            self.last_grip_step = 0
            self.last_grasp = grasp

            # Give some time in between gripper actions
            time.sleep(1.5)

        if self.execute:
            # self.robot.update_desired_ee_pose(
            #     new_target_pose_mat, dt=self.dt
            # )  # , scalar=0.5)
            self.robot.update_desired_ee_pose(
                convert_tip2wrist(new_target_pose_mat), dt=self.dt
            )

        # Draw the current target pose (in meshcat)
        mc_util.meshcat_frame_show(
            self.mc_vis,
            f"scene/target_pose_wrist",
            convert_tip2wrist(new_target_pose_mat.cpu().numpy()),
        )
        mc_util.meshcat_frame_show(
            self.mc_vis, f"scene/target_pose_tip", new_target_pose_mat.cpu().numpy()
        )
        mc_util.meshcat_frame_show(
            self.mc_vis,
            f"scene/current_pose_tip",
            convert_wrist2tip(poly_util.polypose2mat(self.robot.get_ee_pose())),
        )
        mc_util.meshcat_frame_show(
            self.mc_vis,
            f"scene/current_pose_wrist",
            poly_util.polypose2mat(self.robot.get_ee_pose()),
        )

        # calculate timing
        # precise_wait(t_cycle_end)
        self.iter_idx += 1

        self.act_step += 1
        if self.act_step < self.action_horizon:
            # time.sleep(0.01)
            time.sleep(0.025)
        else:
            self.act_step = 0

        # get new obs
        new_obs = self.get_obs()

        return new_obs, None, None, None


def main():
    frequency = args.frequency
    dt = 1 / frequency
    command_latency = args.command_latency

    franka_ip = "173.16.0.1"
    # robot = RobotInterface(ip_address=franka_ip)
    robot = DiffIKWrapper(ip_address=franka_ip)
    gripper = GripperInterface(ip_address=franka_ip)

    # manual home, open gripper first
    gripper.goto(0.08, 0.05, 0.1, blocking=False)
    # robot_home = torch.Tensor([-0.1363, -0.0406, -0.0460, -2.1322, 0.0191, 2.0759, 0.5])
    robot_home = torch.Tensor(
        [-0.0931, 0.0382, 0.1488, -2.3811, -0.0090, 2.4947, 0.1204]
    )
    home_noise = (2 * torch.rand(7) - 1) * np.deg2rad(5)
    robot_home = robot_home + home_noise
    robot.move_to_joint_positions(robot_home)

    Kq_new = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])
    Kqd_new = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])
    robot.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new, adaptive=True)

    zmq_url = f"tcp://127.0.0.1:{args.port_vis}"
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis["scene"].delete()

    # Setup camera streams (via either LCM or pyrealsense)
    # Setup data saving
    eval_save_dir = Path(args.save_dir) / str(datetime.now())
    eval_save_dir.mkdir(exist_ok=True, parents=True)

    episode_dict = {}
    episode_dict["robot_state"] = []
    episode_dict["image_front"] = []
    episode_dict["image_wrist"] = []
    episode_dict["actions"] = []

    # Make the device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    api.flush()
    # Get the run(s) to test
    run: Run = api.run(args.run_id)

    model_file = [
        f for f in run.files() if f.name.endswith(".pt") and args.wts_name in f.name
    ][0]
    model_path = model_file.download(
        root=f"./models/{run.name}", exist_ok=True, replace=True
    ).name

    print(f"Model path: {model_path}")

    # Create the config object with the project name and make it read-only
    inference_steps = 8
    action_horizon = 8
    config: DictConfig = OmegaConf.create(
        {
            **run.config,
            "project_name": run.project,
            "actor": {
                **run.config["actor"],
                "inference_steps": inference_steps,
                "action_horizon": action_horizon,
            },
        },
        # flags={"readonly": True},
    )

    print(OmegaConf.to_yaml(config))

    # Make the actor
    actor: Actor = get_actor(cfg=config, device=device)

    # Load the model weights
    state_dict = torch.load(model_path)

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    if "model._dummy_variable" in state_dict:
        del state_dict["model._dummy_variable"]

    actor.load_state_dict(state_dict)
    actor.eval()
    actor.cuda()

    actor.encoder1.eval()
    actor.encoder2.eval()
    actor.model.eval()

    assert actor.encoder1.training is False
    assert actor.encoder2.training is False
    assert actor.model.training is False

    # mc
    # actor.set_mc(mc_vis)

    part_poses_norm_mid = None
    if args.observation_type == "state":
        part_poses_norm_max = actor.normalizer.stats.parts_poses.max
        part_poses_norm_min = actor.normalizer.stats.parts_poses.min
        part_poses_norm_mid = (part_poses_norm_max + part_poses_norm_min) / 2.0

    # simple env wrapper
    env = SimpleDiffIKFrankaEnv(
        mc_vis=mc_vis,
        robot=robot,
        gripper=gripper,
        use_lcm=args.use_lcm,
        execute=args.execute,
        observation_type=args.observation_type,
        proprioceptive_state_dim=config.robot_state_dim,
        control_mode=config.control.control_mode,
        furniture_name=args.furniture,
        part_poses_norm_mid=part_poses_norm_mid,
    )

    obs = env.get_obs()

    obs_horizon = actor.obs_horizon

    # Resize the images in the observation if they exist
    resize_image(obs, "color_image1")
    resize_crop_image(obs, "color_image2")

    t_start = time.monotonic()

    env.set_start(t_start)
    env.set_timing(iter_idx=0, dt=dt, command_latency=command_latency)

    # Make structs and keyboard interface for easy collect and infer logging if we want it
    keyboard = KeyboardInterface()
    demo_save_dir = None
    if args.log_ci:
        assert (
            args.ci_save_dir is not None
        ), f"Must set args.ci_save_dir to run with args.log_ci True"
        demo_save_dir = Path(args.ci_save_dir)
        demo_save_dir.mkdir(exist_ok=True, parents=True)

        assert (
            args.furniture is not None
        ), f"Must set args.furniture to run with args.log_ci True"
    ci_helper = CollectInferHelper(args=args, demo_save_dir=demo_save_dir)

    print(f"Starting evaluation...\n\n\n")
    time.sleep(1.0)
    running = True

    start_time = time.time()
    while True:
        # Catch keyboard press, and potentially start recording new episode
        _, collect_enum = keyboard.get_action()
        ci_helper.set_recording(collect_enum == CollectEnum.RECORD)

        if args.log_ci and ci_helper.is_recording:
            # log the data
            ci_helper.log(action_pred, obs)

            if collect_enum == CollectEnum.SUCCESS:
                # save recent segment of trajectory
                ci_helper.save_pkl()
                time.sleep(2.0)
            elif collect_enum == CollectEnum.FAIL:
                # new episode, but don't save
                ci_helper.init_episode()
                time.sleep(2.0)
            else:
                pass

        if collect_enum == CollectEnum.PAUSE_HOLD:
            running = not running

        if not running:
            print(f"Pause button pressed")
            time.sleep(1.0)
            continue

        action_pred = actor.action(obs)

        obs, reward, done, _ = env.step(action_pred[0])

        # Resize the images in the observation if they exist
        resize_image(obs, "color_image1")
        resize_crop_image(obs, "color_image2")


if __name__ == "__main__":
    main()
