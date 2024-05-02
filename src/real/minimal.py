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
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from datetime import datetime

# imports for different interfaces to the cameras
from rdt.image.factory import get_realsense_rgbd_subscribers
import lcm
from rdt.image.factory import enable_realsense_devices
import pyrealsense2 as rs

from typing import List

from omegaconf import OmegaConf, DictConfig
from src.behavior.base import Actor  # noqa
from src.behavior import get_actor
from src.dataset import get_normalizer
from src.data_processing.utils import resize, resize_crop

# from src.common.geometry import quat_xyzw_to_rot_6d

import wandb
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


def isaac_quat_to_pytorch3d_quat(quat):
    """Converts IsaacGym quaternion to PyTorch3D quaternion.

    IsaacGym quaternion is (x, y, z, w) while PyTorch3D quaternion is (w, x, y, z).
    """
    return torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)


def quat_xyzw_to_rot_6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Converts IsaacGym quaternion to rotation 6D."""
    # Move the real part from the back to the front
    quat_wxyz = isaac_quat_to_pytorch3d_quat(quat_xyzw)

    # Convert each quaternion to a rotation matrix
    rot_mats = pt.quaternion_to_matrix(quat_wxyz)

    # Extract the first two columns of each rotation matrix
    rot_6d = pt.matrix_to_rotation_6d(rot_mats)

    return rot_6d


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

parser.add_argument("--run-id", type=str, required=False, nargs="*")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--leaderboard", action="store_true")
parser.add_argument("--project-id", type=str, default=None)
parser.add_argument(
    "--action-type", type=str, default="delta", choices=["delta", "pos"]
)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("-ex", "--execute", action="store_true")
parser.add_argument("-s", "--state_only", action="store_true")

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


class SimpleDiffIKFrankaEnv:
    def __init__(
        self,
        mc_vis: meshcat.Visualizer,
        robot: RobotInterface,
        gripper: GripperInterface,
        use_lcm: bool = False,
        device: str = "cuda",
        execute: bool = False,
        state_only: bool = False,
    ):

        self.device = device
        self.execute = execute

        if use_lcm:
            self._setup_lcm()
            self.get_rgbd = self.get_rgbd_lcm
        else:
            self._setup_rs2()
            self.get_rgbd = self.get_rgbd_rs

        self.mc_vis = mc_vis
        self.robot = robot
        self.gripper = gripper
        self.gripper_open = True
        self.last_grasp = torch.Tensor([-1]).float().to(self.device).squeeze()
        self.last_grip_step = 0

        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open and close actions

        # flags
        self.state_only = state_only

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
        serials = rs_cfg.SERIAL_NUMBERS

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

    def get_obs(self):
        # get observations
        # NOTE: In the simulator we get a 14-dimensional vector with proprioception:
        # [pos, quat, vel, ang vel, gripper], while here we only get 7(?)

        rgbd_list = self.get_rgbd()
        if rgbd_list is None:
            raise ValueError("Could not get list of RGB-D images!")
        # Quaternions follow the convention of <x, y, z, w>
        pos, quat_xyzw = self.robot.get_ee_pose()
        # current_joint_positions = self.robot.get_joint_positions()
        # robot_state_dict = dict(
        #     ee_pose=current_ee_pose.numpy(),
        #     joint_positions=current_joint_positions.numpy(),
        # )

        # TODO (maybe): end effector velocity
        # current_joint_pos = robot.get_joint_positions()
        # jacobian = robot.robot_model.compute_jacobian(current_joint_pos)
        # ee_vel = jacobian @ robot.get_joint_velocities()

        # TODO: We also need the gripper state here
        gripper_width = torch.Tensor([self.gripper.get_state().width])

        # rot_6d = quat_xyzw_to_rot_6d(quat_xyzw)
        # robot_state = torch.cat([pos, rot_6d], dim=-1)
        robot_state = torch.cat([pos, quat_xyzw, gripper_width], dim=-1)

        # convert to tensors
        robot_state = robot_state.reshape(1, -1).to(self.device)
        img1_tensor = torch.from_numpy(rgbd_list[0]["rgb"]).unsqueeze(0).to(self.device)
        img2_tensor = torch.from_numpy(rgbd_list[1]["rgb"]).unsqueeze(0).to(self.device)

        if self.state_only:
            obs = dict(
                parts_poses=torch.Tensor([[]]).to(self.device),
                robot_state=robot_state,
            )
        else:
            obs = dict(
                color_image1=img1_tensor,
                color_image2=img2_tensor,
                robot_state=robot_state,
            )
        return obs

    def step(self, action):

        # # calculate timing
        # t_cycle_end = self.t_start + (self.iter_idx + 1) * self.dt
        # t_sample = t_cycle_end - self.command_latency
        # # t_command_target = t_cycle_end + dt
        # precise_wait(t_sample)

        # convert action
        new_target_pose_mat = self.action2pose_mat(action)

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
            else:
                # goto for opening
                self.gripper.goto(0.08, 0.05, 0.1, blocking=False)

            self.gripper_open = not self.gripper_open
            self.last_grip_step = 0
            self.last_grasp = grasp

        if self.execute:
            self.robot.update_desired_ee_pose(
                new_target_pose_mat, dt=self.dt
            )  # , scalar=0.5)

        # Draw the current target pose (in meshcat)
        mc_util.meshcat_frame_show(
            self.mc_vis, f"scene/target_pose", new_target_pose_mat.cpu().numpy()
        )
        mc_util.meshcat_frame_show(
            self.mc_vis,
            f"scene/current_pose",
            poly_util.polypose2mat(self.robot.get_ee_pose()),
        )

        # calculate timing
        # precise_wait(t_cycle_end)
        self.iter_idx += 1
        time.sleep(0.05)

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

    # manual home
    robot_home = torch.Tensor([-0.1363, -0.0406, -0.0460, -2.1322, 0.0191, 2.0759, 0.5])
    robot.move_to_joint_positions(robot_home)

    Kq_new = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])
    Kqd_new = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])
    robot.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new, adaptive=True)

    gripper = GripperInterface(ip_address=franka_ip)
    gripper.goto(0.08, 0.05, 0.1, blocking=False)

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
    run: Run = api.run(f"{args.run_id}")

    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(
        root=f"./models/{run.name}", exist_ok=True, replace=True
    ).name

    print(f"Model path: {model_path}")

    # Create the config object with the project name and make it read-only
    config: DictConfig = OmegaConf.create(
        {
            **run.config,
            "project_name": run.project,
            "actor": {
                **run.config["actor"],
                "inference_steps": 16,
                "action_horizon": 16,
            },
        },
        # flags={"readonly": True},
    )

    # Get the normalizer
    normalizer_type = config.get("data", {}).get("normalization", "min_max")
    normalizer = get_normalizer(
        normalizer_type=normalizer_type,
        control_mode=config.control.control_mode,
    )

    print(OmegaConf.to_yaml(config))

    # Make the actor
    actor: Actor = get_actor(cfg=config, normalizer=normalizer, device=device)

    # Load the model weights
    state_dict = torch.load(model_path)

    actor.load_state_dict(state_dict)
    actor.eval()
    actor.cuda()

    # mc
    actor.set_mc(mc_vis)

    # simple env wrapper
    env = SimpleDiffIKFrankaEnv(
        mc_vis=mc_vis,
        robot=robot,
        gripper=gripper,
        use_lcm=args.use_lcm,
        execute=args.execute,
        state_only=args.state_only,
    )

    obs = env.get_obs()

    obs_horizon = actor.obs_horizon

    # Resize the images in the observation if they exist
    resize_image(obs, "color_image1")
    resize_crop_image(obs, "color_image2")

    # keep a queue of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon,
        maxlen=obs_horizon,
    )

    t_start = time.monotonic()

    env.set_start(t_start)
    env.set_timing(iter_idx=0, dt=dt, command_latency=command_latency)

    while True:

        # Get the next actions from the actor
        action_pred = actor.action(obs_deque)

        obs, reward, done, _ = env.step(action_pred[0])

        # NOTE: Should implement storage of rollouts for further training?
        # video_obs = obs.copy()

        # Resize the images in the observation if they exist
        resize_image(obs, "color_image1")
        resize_crop_image(obs, "color_image2")

        # Save observations for the policy
        obs_deque.append(obs)


if __name__ == "__main__":
    main()
