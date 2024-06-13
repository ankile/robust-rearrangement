# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import argparse
import time
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from omni.isaac.kit import SimulationApp

# from furniture_bench.utils.pose import get_mat, rot_mat

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Force display off at all times.",
)
parser.add_argument("-i", "--demo-index", type=int, default=0)
parser.add_argument("-sub", "--sub-steps", type=int, default=0)
parser.add_argument("--load-dir", type=str, required=True)
parser.add_argument("--save", action="store_true")
parser.add_argument("--save-dir", type=str, default=None)
parser.add_argument("--num-parts", type=int, required=True)
parser.add_argument("--furniture", type=str, required=True)
parser.add_argument("--visualize-actions", action="store_true")
parser.add_argument("-dr", "--domain-rand", action="store_true")

# Create a subparser for randomization arguments
subparsers = parser.add_subparsers(dest="subcommand")

# Create the parser for the "rand" command
rand_parser = subparsers.add_parser("rand", help="Randomization arguments")
rand_parser.add_argument(
    "--part-random", type=str, default="base", help="Randomize part colors"
)
rand_parser.add_argument(
    "--table-random", type=str, default="base", help="Randomize table colors"
)
rand_parser.add_argument(
    "--different-part-colors", action="store_true", help="Use different part colors"
)
rand_parser.add_argument(
    "--random-frame-freq", type=int, default=1, help="Randomize every X frames"
)

args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp(
    {
        "headless": args_cli.headless,
        "width": 640,
        "height": 480,
        # "width": int(640 * 2),
        # "height": int(480 * 2),
    }
)
"""Rest everything follows."""

import numpy as np
import torch
import carb
from torchvision.transforms import functional as F, InterpolationMode
from pxr import Gf
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

from pxr import UsdShade
from omni.isaac.debug_draw import _debug_draw

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
import omni.replicator.core as rep
from omni.isaac.orbit.utils import convert_dict_to_backend

import furniture_bench.utils.transform as T
from furniture_bench.config import config

from src.sim2real.part_config_render import part_config_dict

from typing import Dict, Union, Tuple

# folder to load from
DEMO_DIR = args_cli.load_dir
FILES = [
    os.path.join(DEMO_DIR, f)
    for f in os.listdir(DEMO_DIR)
    if f.endswith(".pkl") and "base_action" not in f
]
FILE = FILES[args_cli.demo_index]

PART_COLOR_BASE = (
    "white" if "PART_COLOR_BASE" not in os.environ else os.environ["PART_COLOR_BASE"]
)

# constants
april_to_sim_mat = np.array(
    [
        [6.1232343e-17, 1.0000000e00, 1.2246469e-16, 1.4999807e-03],
        [1.0000000e00, -6.1232343e-17, -7.4987988e-33, 0.0000000e00],
        [0.0000000e00, 1.2246469e-16, -1.0000000e00, 4.1500002e-01],
        [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
    ]
)
front_435_camera_matrix = np.array(
    [
        [607.24603271, 0.0, 328.35696411],
        [0.0, 607.39056396, 244.84118652],
        [0.0, 0.0, 1.0],
    ]
)
wrist_435_camera_matrix = np.array(
    [
        [613.14752197, 0.0, 326.19647217],
        [0.0, 613.16229248, 244.59855652],
        [0.0, 0.0, 1.0],
    ]
)

draw = _debug_draw.acquire_debug_draw_interface()


def build_camera_params(
    width: int = 640, height: int = 480, pixel_size: float = (3.0 * 1e-3)
) -> Dict[str, Dict[str, float]]:

    cam_params = {}
    for name, mat in zip(
        ["front", "wrist"], [front_435_camera_matrix, wrist_435_camera_matrix]
    ):
        camera_matrix = mat
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix

        focal_length_x = fx * pixel_size
        focal_length_y = fy * pixel_size

        horizontal_aperture = pixel_size * width
        vertical_aperture = pixel_size * height

        params = dict(
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            focal_length=(focal_length_x + focal_length_y) / 2,
            f_stop=200.0 if name == "front" else 0.0,
            focus_distance=0.6 if name == "front" else 0.0,
        )
        cam_params[name] = params

    return cam_params


def run_until_quit(simulation_app, world):
    import omni.appwindow
    import carb.input

    global RP_RUNNING, RP_PAUSE, return_flag
    RP_RUNNING = True
    RP_PAUSE = False
    return_flag = None

    def on_kb_input(e):
        global RP_RUNNING, RP_PAUSE, return_flag
        # if e.type == carb.input.KeyboardEventType.KEY_RELEASE:
        # if e.type == carb.input.KeyboardEventType.KEY_PRESS:
        if (
            e.type == carb.input.KeyboardEventType.KEY_REPEAT
            or e.type == carb.input.KeyboardEventType.KEY_RELEASE
        ):
            if e.input == carb.input.KeyboardInput.Q:
                print(f'[Run until quit] Caught "Q" keypress, closing app')
                RP_RUNNING = False
                return_flag = 2
            if e.input == carb.input.KeyboardInput.F:
                print(f'[Run until quit] Caught "Q" keypress, closing app')
                RP_RUNNING = False
                return_flag = 1
            elif e.input == carb.input.KeyboardInput.B:
                print(f'[Run until quit] Caught "Q" keypress, closing app')
                RP_RUNNING = False
                return_flag = -1
            elif e.input == carb.input.KeyboardInput.P:
                print(f'[Run until quit] Caught "P" keypress, pausing')
                RP_PAUSE = True

    app_window = omni.appwindow.get_default_app_window()
    kb = app_window.get_keyboard()
    carb_input = carb.input.acquire_input_interface()
    _ = carb_input.subscribe_to_keyboard_events(kb, on_kb_input)

    while RP_RUNNING:
        # simulation_app.update()
        world.render()
        time.sleep(1 / 120.0)

        if RP_PAUSE:
            print(f"[Run until quit] Paused!")
            from IPython import embed

            embed()
            RP_PAUSE = False

    return return_flag


def gen_prim(usd_path: str, prim_name: str, init_pos, init_ori):
    furniture_assets_root_path = Path(
        f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd"
    )
    # furniture_assets_root_path = Path(f"assets/furniture/mesh/usd")
    usd_path = str(furniture_assets_root_path / usd_path)
    assert os.path.exists(usd_path), f"Could not find USD file: {usd_path}"

    pose = april_to_sim_mat @ (T.to_homogeneous(init_pos, init_ori))
    pos, ori = T.mat2pose(pose)
    ori = T.convert_quat(ori, to="wxyz")

    prim_utils.create_prim(
        prim_name, usd_path=usd_path, translation=pos, orientation=ori
    )

    view = XFormPrimView(prim_name, reset_xform_properties=False)
    return view


light_prim_paths = [
    "/World/Table/Lights/RectLight",
    "/World/Table/Lights/RectLight_01",
    "/World/Table/Lights/CylinderLight",
    "/World/Table/Lights/CylinderLight_01",
    "/World/Table/Lights/CylinderLight_02",
    "/World/Table/Lights/CylinderLight_03",
]


part_mat_paths = [
    "/World/ObstacleFront/Looks/OmniPBR_ClearCoat",
    "/World/ObstacleRight/Looks/OmniPBR_ClearCoat",
    "/World/ObstacleLeft/Looks/OmniPBR_ClearCoat",
]


table_mat_paths = [
    "/World/Table/Looks/Material_010",
]

camera_prim_paths = ["/World/CameraSensor"]


def rnd_high_low(high: np.ndarray, low: np.ndarray):
    rnd = np.random.random(high.shape[0])
    return low + (high - low) * rnd


def rnd_about_nominal(
    nom: np.ndarray,
    dist_type: str = "gaussian",
    uniform_scale: float = None,
    variance: float = None,
):
    assert dist_type in ["gaussian", "uniform"], f"Unrecognized dist type: {dist_type}"
    if dist_type == "gaussian":
        assert variance is not None, f"Must set variance if using Gaussian"
    if dist_type == "uniform":
        assert uniform_scale is not None, f"Must set uniform_scale if using uniform"

    if dist_type == "gaussian":
        rnd = np.random.normal(scale=np.sqrt(variance), size=nom.shape)
    elif dist_type == "uniform":
        rnd = (np.random.random(size=nom.shape) - 0.5) * uniform_scale

    return nom + rnd


def rnd_isaac_pose_about_nominal(
    nom_pose: Tuple[torch.Tensor, torch.Tensor],
    dist_type: str = "gaussian",
    uniform_scale_xyz: float = None,
    uniform_scale_rpyaw: float = None,
    variance_xyz: float = None,
    variance_rpyaw: float = None,
):
    assert dist_type in ["gaussian", "uniform"], f"Unrecognized dist type: {dist_type}"
    if dist_type == "gaussian":
        assert variance_xyz is not None, f"Must set variance if using Gaussian"
        assert variance_rpyaw is not None, f"Must set variance if using Gaussian"
    if dist_type == "uniform":
        assert uniform_scale_xyz is not None, f"Must set uniform_scale if using uniform"
        assert (
            uniform_scale_rpyaw is not None
        ), f"Must set uniform_scale if using uniform"

    if dist_type == "gaussian":
        rnd_xyz = torch.normal(mean=0.0, std=np.sqrt(variance_xyz), size=(3,))
        rnd_rpyaw = torch.normal(mean=0.0, std=np.sqrt(variance_rpyaw), size=(3,))
    elif dist_type == "uniform":
        rnd_xyz = (torch.rand(size=(3,)) - 0.5) * uniform_scale_xyz
        rnd_rpyaw = (torch.rand(size=(3,)) - 0.5) * uniform_scale_rpyaw

    # add the position part
    nom_pos = nom_pose[0]
    new_pos = nom_pos + rnd_xyz

    # convert orientation part properly
    nom_quat = T.convert_quat(nom_pose[1], to="xyzw")
    rnd_quat = T.euler2quat(rnd_rpyaw)
    new_quat = T.quat_multiply(nom_quat, rnd_quat)
    new_quat = T.convert_quat(new_quat, to="wxyz")

    return new_pos, new_quat


def np2Vec3f(arr):
    assert arr.shape[0] == 3, f"Must be 3D to make Vec3f"
    vec = Gf.Vec3f()
    for i, val in enumerate(arr):
        vec[i] = val
    return vec


def resize(img: Union[np.ndarray, torch.Tensor]):
    """Resizes `img` into ..."""
    th, tw = 240, 320
    # th, tw = 480, 640
    was_numpy = False

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        was_numpy = True

    if isinstance(img, torch.Tensor):
        # Move channels in front (B, H, W, C) -> (B, C, H, W)
        if len(img.shape) == 4:
            img = img.permute(0, 3, 1, 2)
        else:
            img = img.permute(2, 0, 1)

    img = F.resize(
        img, (th, tw), interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    if isinstance(img, torch.Tensor):
        # Move channels back (B, C, H, W) -> (B, H, W, C)
        if len(img.shape) == 4:
            img = img.permute(0, 2, 3, 1)
        else:
            img = img.permute(1, 2, 0)

    if was_numpy:
        img = img.numpy()

    return img


def resize_dict(img_dict: dict):
    img_dict["rgb"] = resize(img_dict["rgb"][:, :, :3])
    return img_dict


@dataclass
class RandomizationConfig:
    part_random: str = "base"
    table_random: str = "base"
    different_part_colors: bool = False
    random_frame_freq: int = 1


class RandomizationHelper:
    def __init__(self, rand_config: RandomizationConfig):
        self.rand_config = rand_config

        assert self.rand_config.part_random in ["base", "full"]
        self.part_random = self.rand_config.part_random

        assert self.rand_config.table_random in ["base", "full"]
        self.table_random = self.rand_config.table_random

        self.different_part_colors = self.rand_config.different_part_colors

        self._setup()

    def _setup(self):

        # get the nominal color for the table

        table_mat_prim = prim_utils.get_prim_at_path(table_mat_paths[0])
        # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
        assert (
            table_mat_prim.IsValid()
        ), f"Prim {table_mat_prim} at path {table_mat_paths[0]} is not valid"
        table_shader_prim = table_mat_prim.GetChildren()[0]
        self.nominal_color = np.asarray(
            table_shader_prim.GetAttribute("inputs:diffuse_color_constant").Get()
        )

        # set different low levels for part colors
        self.part_low_high = np.array([0.984, 0.889, 0.843])
        self.part_low_middle = np.array([0.5, 0.5, 0.5])

        self.part_color_low = (
            self.part_low_middle if self.part_random == "full" else self.part_low_high
        )

        self.part_high_middle = np.array([0.5, 0.5, 0.5])
        self.part_high_low = np.array([0.02, 0.02, 0.02])

        self.part_color_high = (
            self.part_high_middle if self.part_random == "full" else self.part_high_low
        )

        self.global_cams = []
        self.global_cam_poses = []
        self.local_cams = []
        self.local_cam_poses = []

    def set_global_cams(self, global_cams):
        self.global_cams = global_cams
        self.global_cam_poses = [
            cam._sensor_xform.get_world_pose() for cam in self.global_cams
        ]

    def set_local_cams(self, local_cams):
        self.local_cams = local_cams
        self.local_cam_poses = [
            cam._sensor_xform.get_local_pose() for cam in self.local_cams
        ]

    def random_table_colors(self):
        # set table color rand function
        if self.table_random == "full":
            return self.random_table_colors_full()
        else:
            return self.random_table_colors_nominal()

    def toggle_lights(self):
        if np.random.random() > 0.75:
            return
        for lpp in light_prim_paths:
            light_prim = prim_utils.get_prim_at_path(lpp)
            # print(f"Light prim: {light_prim} at path: {lpp}")
            assert light_prim.IsValid(), f"Prim {light_prim} at path {lpp} is not valid"
            visible_attr = light_prim.GetAttribute("visibility")

            if np.random.random() > 0.5:
                visible_attr.Set("invisible")
            else:
                visible_attr.Set("inherited")

    def random_light_colors(self):
        high = np.array([1.0, 1.0, 1.0])
        low = np.array([0.789, 0.715, 0.622])
        for lpp in light_prim_paths:
            # print(f"Light prim: {light_prim} at path: {lpp}")
            light_prim = prim_utils.get_prim_at_path(lpp)
            assert light_prim.IsValid(), f"Prim {light_prim} at path {lpp} is not valid"
            color_attr = light_prim.GetAttribute("color")
            color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
            color_attr.Set(color_to_set_gf)

    def random_light_intensity(self):
        high = np.array([8000])
        low = np.array([3000])
        for lpp in light_prim_paths:
            # print(f"Light prim: {light_prim} at path: {lpp}")
            light_prim = prim_utils.get_prim_at_path(lpp)
            assert light_prim.IsValid(), f"Prim {light_prim} at path {lpp} is not valid"
            intensity_attr = light_prim.GetAttribute("intensity")
            intensity_attr.Set(rnd_high_low(high, low)[0])

    def random_part_colors(self):

        if PART_COLOR_BASE == "white":
            high = np.array([1.0, 1.0, 1.0])
            # low = np.array([0.984, 0.889, 0.843])
            # low = np.array([0.5, 0.5, 0.5])
            low = self.part_color_low
        elif PART_COLOR_BASE == "black":
            high = self.part_color_high
            low = np.array([0.0, 0.0, 0.0])
        color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
        for pmpp in part_mat_paths:
            part_mat_prim = prim_utils.get_prim_at_path(pmpp)
            # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
            assert (
                part_mat_prim.IsValid()
            ), f"Prim {part_mat_prim} at path {pmpp} is not valid"
            shader_prim = part_mat_prim.GetChildren()[0]
            color_attr = shader_prim.GetAttribute("inputs:diffuse_color_constant")
            if self.different_part_colors:
                color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
            color_attr.Set(color_to_set_gf)

    def random_table_colors_nominal(self):

        table_mat_prim = prim_utils.get_prim_at_path(table_mat_paths[0])
        # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
        assert (
            table_mat_prim.IsValid()
        ), f"Prim {table_mat_prim} at path {table_mat_paths[0]} is not valid"
        table_shader_prim = table_mat_prim.GetChildren()[0]

        # nominal_color = np.asarray(
        #     table_shader_prim.GetAttribute("inputs:diffuse_color_constant").Get()
        # )
        nominal_color = self.nominal_color
        dist_type = "gaussian"
        variance = 0.0001

        color_attr = table_shader_prim.GetAttribute("inputs:diffuse_color_constant")
        color_to_set_gf = np2Vec3f(
            rnd_about_nominal(nom=nominal_color, dist_type=dist_type, variance=variance)
        )
        color_attr.Set(color_to_set_gf)

    def random_table_colors_full(self):

        table_mat_prim = prim_utils.get_prim_at_path(table_mat_paths[0])
        # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
        assert (
            table_mat_prim.IsValid()
        ), f"Prim {table_mat_prim} at path {table_mat_paths[0]} is not valid"
        table_shader_prim = table_mat_prim.GetChildren()[0]

        high = np.array([1.0, 1.0, 1.0])
        low = np.array([0.5, 0.5, 0.5])

        color_attr = table_shader_prim.GetAttribute("inputs:diffuse_color_constant")
        color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
        color_attr.Set(color_to_set_gf)

    def random_camera_pose(self):

        dist_type = "gaussian"
        std_xyz = 0.0025
        std_rpyaw = np.deg2rad(0.5)
        for i, cam in enumerate(self.global_cams):
            nom_world_pose = self.global_cam_poses[i]
            new_world_pose = rnd_isaac_pose_about_nominal(
                nom_pose=nom_world_pose,
                dist_type=dist_type,
                variance_xyz=std_xyz**2,
                variance_rpyaw=std_rpyaw**2,
            )
            new_pos, new_quat = new_world_pose
            cam._sensor_xform.set_world_pose(new_pos, new_quat)

        std_xyz = 0.00075
        std_rpyaw = np.deg2rad(0.25)
        for i, cam in enumerate(self.local_cams):
            nom_local_pose = self.local_cam_poses[i]
            new_local_pose = rnd_isaac_pose_about_nominal(
                nom_pose=nom_local_pose,
                dist_type=dist_type,
                variance_xyz=std_xyz**2,
                variance_rpyaw=std_rpyaw**2,
            )
            new_pos, new_quat = new_local_pose
            cam._sensor_xform.set_local_pose(new_pos, new_quat)


def main():
    s = time.time()
    with open(FILE, "rb") as f:
        data = pickle.load(f)

    base_action_file = FILE.replace(".pkl", "_base_action_pos.pkl")
    if os.path.exists(base_action_file):
        with open(base_action_file, "rb") as f:
            base_action_data = pickle.load(f)
    else:
        base_action_data = None
    """Spawns a single arm manipulator and applies random joint commands."""

    # Load kit helper
    # sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    sim = SimulationContext(physics_dt=0.0001, rendering_dt=0.0001, backend="torch")

    # get camera params
    cam_params = build_camera_params()

    # Setup camera sensor (front)
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=640,
        # width=1440,
        # height=1080,
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=cam_params["front"]["focal_length"],
            horizontal_aperture=cam_params["front"]["horizontal_aperture"],
            clipping_range=(0.1, 1.0e5),
            f_stop=cam_params["front"]["f_stop"],
            focus_distance=cam_params["front"]["focus_distance"],
        ),
    )
    camera = Camera(cfg=camera_cfg, device="cpu")

    # Spawn camera
    camera.spawn("/World/CameraSensor")

    # Setup camera sensor (wrist)
    wrist_camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=640,
        # width=1440,
        # height=1080,
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=cam_params["wrist"]["focal_length"],
            horizontal_aperture=cam_params["wrist"]["horizontal_aperture"],
            clipping_range=(1.0e-5, 1.0e5),
            f_stop=cam_params["wrist"]["f_stop"],
            focus_distance=cam_params["wrist"]["focus_distance"],
        ),
    )
    wrist_camera = Camera(cfg=wrist_camera_cfg, device="cpu")

    # Table
    usd_base_path = Path(
        f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd"
    )
    # usd_base_path = Path(f"assets/furniture/mesh/usd")
    table_z_offset = 0.415

    # prim_utils.create_prim("/World/Table", usd_path=str(usd_base_path / "table.usda"))
    prim_utils.create_prim(
        "/World/Table", usd_path=str(usd_base_path / "table_room.usda")
    )
    table_view = XFormPrimView("/World/Table", reset_xform_properties=False)
    table_view.set_local_scales(torch.Tensor([[1.7, 1.35, 1.0]]))

    views = []

    # Background
    prim_utils.create_prim(
        "/World/Background",
        translation=(-0.8, 0, 0.75),
        usd_path=str((usd_base_path / "background.usda")),
    )

    # Obstacle front.
    obs_quat_wxyz = np.array([0.7071067690849304, 0, 0, 0.7071067690849304])
    prim_utils.create_prim(
        "/World/ObstacleFront",
        translation=(0.3815, 0.0, table_z_offset),
        orientation=obs_quat_wxyz,
        usd_path=str((usd_base_path / "obstacle_front.usda")),
    )

    # Obstacle side.
    prim_utils.create_prim(
        "/World/ObstacleRight",
        translation=(0.306, -0.175, table_z_offset),
        orientation=obs_quat_wxyz,
        usd_path=str((usd_base_path / "obstacle_side.usda")),
    )
    prim_utils.create_prim(
        "/World/ObstacleLeft",
        translation=(0.306, 0.175, table_z_offset),
        orientation=obs_quat_wxyz,
        usd_path=str((usd_base_path / "obstacle_side.usda")),
    )
    obstacle_center_view = XFormPrimView(
        "/World/ObstacleFront", reset_xform_properties=False
    )
    obstacle_right_view = XFormPrimView(
        "/World/ObstacleRight", reset_xform_properties=False
    )
    obstacle_left_view = XFormPrimView(
        "/World/ObstacleLeft", reset_xform_properties=False
    )

    # Parts
    part_config = part_config_dict[args_cli.furniture]

    furn_name = part_config["furniture"]
    for i, part_name in enumerate(part_config["names"]):
        prim_path = part_config["prim_paths"][i]
        usd_path = furn_name + "/" + part_config["usd_names"][i]
        pos = config["furniture"][furn_name][part_name]["reset_pos"][0]
        ori = config["furniture"][furn_name][part_name]["reset_ori"][0][:3, :3]
        view = gen_prim(usd_path, prim_path, pos, ori)
        views.append(view)

    # Robots
    # -- Resolve robot config from command-line arguments
    robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG

    # -- Spawn robot
    robot = SingleArmManipulator(cfg=robot_cfg)
    franka_from_origin_mat = np.array(
        [
            [1.0, 0.0, 0.0, -0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, table_z_offset],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pos, ori = T.mat2pose(franka_from_origin_mat)
    robot.spawn("/World/Robot_2", translation=pos)

    # Spawn wrist camera (has to come after spawning the robot)
    wrist_camera.spawn(
        "/World/Robot_2/panda_hand",
        translation=np.array([-0.05, 0.0, 0.04]),
        orientation=T.convert_quat(
            R.from_euler("XYZ", np.deg2rad([180, -16.5, 90])).as_quat(), to="wxyz"
        ),
    )

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/Robot.*")
    # Reset states
    robot.reset_buffers()

    # Initialize camera
    camera.initialize()

    cam_pos = (0.82, -0.065, 0.8)
    cam_quat = T.convert_quat(
        R.from_euler("XYZ", np.deg2rad([0.0, 68.5, 90])).as_quat(), to="wxyz"
    )
    camera._sensor_xform.set_world_pose(cam_pos, cam_quat)

    wrist_camera.initialize()

    # Create replicator writer
    demo_date = FILE.split("/")[-1].replace(".pkl", "")
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "output",
        "camera",
        # f"{demo_date}_substeps_{args_cli.sub_steps}",
        f"{demo_date}_substeps_{args_cli.sub_steps}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    )
    if args_cli.domain_rand:
        output_dir += "_domain_rand"

    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    wrist_output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "output",
        "wrist_camera",
        # f"{demo_date}_substeps_{args_cli.sub_steps}",
        f"{demo_date}_substeps_{args_cli.sub_steps}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    )
    if args_cli.domain_rand:
        wrist_output_dir += "_domain_rand"

    wrist_rep_writer = rep.BasicWriter(output_dir=wrist_output_dir, frame_padding=3)

    rp3 = rep.create.render_product("/OmniverseKit_Persp", (640, 480))
    rp_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rp_rgb.attach(rp3)
    viewer_rep_writer = rep.BasicWriter(
        output_dir=output_dir + "_viewer", frame_padding=3
    )

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    ep_step_count = 0

    # sim_steps = 1
    sim_steps = 10
    print(f"Sim steps: {sim_steps}")

    prev_goal_pos = torch.tensor(
        data["observations"][0]["robot_state"]["joint_positions"]
    )

    parts_prev_goal_pos = []
    parts_prev_goal_ori = []
    part_idx_offset = 1

    fj1 = data["observations"][0]["robot_state"]["gripper_finger_1_pos"]
    fj2 = data["observations"][0]["robot_state"]["gripper_finger_2_pos"]
    prev_fj1 = data["observations"][0]["robot_state"]["gripper_finger_1_pos"]
    prev_fj2 = data["observations"][0]["robot_state"]["gripper_finger_2_pos"]

    # Initialize robot state
    robot.set_dof_state(
        torch.concat(
            [
                torch.tensor(data["observations"][0]["robot_state"]["joint_positions"]),
                torch.from_numpy(
                    fj1.reshape(
                        1,
                    )
                ).float(),
                torch.from_numpy(
                    fj2.reshape(
                        1,
                    )
                ).float(),
            ]
        ),
        torch.concat(
            [
                torch.tensor(
                    data["observations"][0]["robot_state"]["joint_velocities"]
                ),
                torch.zeros((1,)),
                torch.zeros((1,)),
            ]
        ),
    )

    # initialize parts
    for i in range(args_cli.num_parts):
        pos = data["observations"][part_idx_offset]["parts_poses"][7 * i : 7 * i + 3]
        ori = data["observations"][part_idx_offset]["parts_poses"][
            7 * i + 3 : 7 * i + 7
        ]
        parts_prev_goal_pos.append(pos)
        parts_prev_goal_ori.append(ori)

    # initialize obstacles
    obstacle_center_pose = data["observations"][1]["parts_poses"][-7:]
    obstacle_right_pose = obstacle_center_pose.copy()
    obstacle_left_pose = obstacle_center_pose.copy()

    for obs_name, obs_view, obs_pose in zip(
        ["center", "right", "left"],
        [obstacle_center_view, obstacle_right_view, obstacle_left_view],
        [obstacle_center_pose, obstacle_right_pose, obstacle_left_pose],
    ):

        rot = T.quat2mat(obs_pose[3:])
        obs_pose_world = april_to_sim_mat @ (T.to_homogeneous(obs_pose[:3], rot))
        obs_pose_pos, obs_pose_ori = T.mat2pose(obs_pose_world)
        obs_pose_ori = T.convert_quat(obs_pose_ori, to="wxyz")

        if obs_name == "right":
            obs_pose_pos[0] -= 0.075
            obs_pose_pos[1] -= 0.175
        elif obs_name == "left":
            obs_pose_pos[0] -= 0.075
            obs_pose_pos[1] += 0.175

        obs_pose_pos = torch.from_numpy(obs_pose_pos).unsqueeze(0)
        obs_pose_ori = torch.from_numpy(obs_pose_ori).unsqueeze(0)

        obs_view.set_world_poses(positions=obs_pose_pos, orientations=obs_pose_ori)

    for _ in range(100):
        sim.step()
        time.sleep(0.01)

    # setup re-saving
    if args_cli.save:
        assert args_cli.save_dir is not None, f"Must set --save-dir if --save is True!"
        demo_save_dir = Path(args_cli.save_dir)
        demo_save_dir.mkdir(exist_ok=True, parents=True)
        pkl_path = demo_save_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

    episode_data = {}
    episode_data["observations"] = []
    episode_data["actions"] = data["actions"]
    episode_data["furniture"] = data["furniture"]
    # assume all real world demos that we actually save are success
    episode_data["success"] = True
    # episode_data["args"] = data["args"]
    episode_data["args_rerender"] = args_cli.__dict__
    if "metadata" in data:
        episode_data["metadata"] = data["metadata"]
    else:
        episode_data["metadata"] = None

    fps_list = []
    last_time = time.time()

    # Use the subcommand arguments
    dr_config = RandomizationConfig()
    if args_cli.subcommand == "rand":
        rand_args = args_cli
        for rand_key in dr_config.__dict__.keys():
            print(f"Rand key: {rand_key}, args value: {rand_args.__dict__[rand_key]}")
            dr_config.__dict__[rand_key] = rand_args.__dict__[rand_key]

    dr_helper = RandomizationHelper(rand_config=dr_config)
    dr_helper.set_global_cams([camera])
    dr_helper.set_local_cams([wrist_camera])

    # Try to automatically retrieve all the material prims
    for i, prim_path in enumerate(part_config["prim_paths"]):
        looks_prim_path = prim_path + "/Looks"
        assert prim_utils.is_prim_path_valid(
            looks_prim_path
        ), f"Could not find Looks prim: {looks_prim_path}"
        looks_prim = prim_utils.get_prim_at_path(looks_prim_path)
        for looks_child_prim in looks_prim.GetChildren():
            looks_child_path = looks_child_prim.GetPath().pathString
            if (
                "Material" in looks_child_path
                and looks_child_path not in part_mat_paths
            ):
                print(
                    f"Adding Material path: {looks_child_path} to materials for randomization"
                )
                part_mat_paths.append(looks_child_path)

    print(f"Found material prims for all parts: {part_mat_paths}")

    # Drawing helpers

    def make_part_clear():
        plastic_clear_mat = UsdShade.Material(
            prim_utils.get_prim_at_path("/World/Table/Looks/Plastic_Clear")
        )
        leg_prim = prim_utils.get_prim_at_path("/World/SquareTableLeg4")
        UsdShade.MaterialBindingAPI(leg_prim).Bind(
            plastic_clear_mat, UsdShade.Tokens.strongerThanDescendants
        )

    # def make_lines_start_ee_pos(ee_pos, ee_quat, delta_vector, scale=1.0):
    def make_lines_start_ee_pos(
        start_pos, ee_quat, delta_vector, scale=1.0, with_offset=True
    ):

        # Random start near the EE pos
        # noise = (np.random.random(3) - 0.5).astype(np.float32).reshape(1, 3) * 0.001
        offset = (
            franka_from_origin_mat[:-1, -1].reshape(1, 3)
            if with_offset
            else np.zeros((1, 3))
        )
        # line_start = ee_pos.reshape(1, 3) + offset + noise
        line_start = start_pos.reshape(1, 3) + offset

        # Move the start point higher
        ee_z_axis = T.quat2mat(ee_quat)[:, 2].reshape(1, 3)
        # line_start = line_start - ee_z_axis * 0.019

        line_end = line_start + delta_vector * scale
        lines = np.concatenate([line_start, line_end], axis=0)
        return lines

    def draw_res_action(ee_pos, ee_quat, base_action_pos, full_action_pos):
        scale = 2.0
        # scale = 1.0

        # base action first
        base_delta_pos = base_action_pos - ee_pos
        lines = make_lines_start_ee_pos(
            ee_pos, ee_quat, delta_vector=base_delta_pos, scale=scale
        )
        colors = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32) * 255
        sizes = [3.0]
        draw.draw_lines(
            [carb._carb.Float3(lines[0])],
            [carb._carb.Float3(lines[1])],
            [carb._carb.ColorRgba(colors)],
            sizes,
        )
        # draw.draw_lines(carb._carb.Float3(lines[0]), carb._carb.Float3(lines[1]), carb._carb.ColorRgba(colors), sizes)
        # draw.draw_lines(lines.tolist(), colors.tolist(), sizes)

        # residual action
        residual_delta_pos = full_action_pos - base_action_pos
        # lines = make_lines_start_ee_pos(
        #     ee_pos, ee_quat, delta_vector=residual_delta_pos, scale=scale
        # )
        lines = make_lines_start_ee_pos(
            lines[1],
            ee_quat,
            delta_vector=residual_delta_pos,
            scale=scale,
            with_offset=False,
        )
        colors = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32) * 255
        sizes = [3.0]
        draw.draw_lines(
            [carb._carb.Float3(lines[0])],
            [carb._carb.Float3(lines[1])],
            [carb._carb.ColorRgba(colors)],
            sizes,
        )

        # full action
        full_delta_pos = full_action_pos - ee_pos
        lines = make_lines_start_ee_pos(
            ee_pos, ee_quat, delta_vector=full_delta_pos, scale=scale
        )
        colors = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32) * 255
        # colors = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32) * 255
        sizes = [3.0]
        draw.draw_lines(
            [carb._carb.Float3(lines[0])],
            [carb._carb.Float3(lines[1])],
            [carb._carb.ColorRgba(colors)],
            sizes,
        )

    # run_until_quit(simulation_app=simulation_app, world=sim)

    if args_cli.visualize_actions:
        make_part_clear()

    # obs_idx = 0
    obs_idx = 100
    pause_idx = 110  # 130
    run_return_flag = 1

    if (
        base_action_data is not None
        and args_cli.visualize_actions
        and obs_idx < len(base_action_data)
    ):
        # see if we should visualize
        obs = data["observations"][obs_idx]
        base_action_pos = base_action_data[obs_idx]
        full_action_pos = obs["robot_state"]["ee_pos"] + data["actions"][obs_idx][:3]
        prev_base_action_pos = base_action_pos
        prev_full_action_pos = full_action_pos
        prev_ee_pos = obs["robot_state"]["ee_pos"]
        prev_ee_quat = obs["robot_state"]["ee_quat"]

    while True:
        # for obs_idx, obs in enumerate(data["observations"]):
        obs = data["observations"][obs_idx]

        if args_cli.domain_rand:
            if obs_idx % dr_config.random_frame_freq == 0:
                dr_helper.toggle_lights()
                dr_helper.random_light_colors()
                dr_helper.random_light_intensity()
                dr_helper.random_part_colors()
                dr_helper.random_table_colors()
                dr_helper.random_camera_pose()

        goal_pos = torch.tensor(obs["robot_state"]["joint_positions"])
        dx = (goal_pos - prev_goal_pos) / sim_steps

        fj1 = obs["robot_state"]["gripper_finger_1_pos"]
        fj2 = obs["robot_state"]["gripper_finger_2_pos"]
        fj1_dx = (fj1 - prev_fj1) / sim_steps
        fj2_dx = (fj2 - prev_fj2) / sim_steps

        part_idx = (
            obs_idx + part_idx_offset
            if obs_idx + part_idx_offset < len(data["observations"])
            else len(data["observations"]) - 1
        )

        # if obs_idx == 230:
        #     # if obs_idx == 50:
        #     run_until_quit(simulation_app=simulation_app, world=sim)
        #     from IPython import embed

        #     embed()
        #     assert False

        if (
            base_action_data is not None
            and args_cli.visualize_actions
            and obs_idx < len(base_action_data)
        ):
            # see if we should visualize
            base_action_pos = base_action_data[obs_idx]
            full_action_pos = (
                obs["robot_state"]["ee_pos"] + data["actions"][obs_idx][:3]
            )
            ee_pos = obs["robot_state"]["ee_pos"]
            ee_quat = obs["robot_state"]["ee_quat"]

        for i in range(int(sim_steps)):
            interp_goal = prev_goal_pos + (i + 1) * dx
            interp_fj1 = prev_fj1 + (i + 1) * fj1_dx
            interp_fj2 = prev_fj2 + (i + 1) * fj2_dx

            # If simulation is paused, then skip.
            if not sim.is_playing():
                sim.step(render=not args_cli.headless)
                continue

            # Update camera data
            camera.update(dt=0.0)
            wrist_camera.update(dt=0.0)

            if not args_cli.save:
                rep_writer.write(
                    convert_dict_to_backend(camera.data.output, backend="numpy")
                )
                wrist_rep_writer.write(
                    convert_dict_to_backend(wrist_camera.data.output, backend="numpy")
                )
                viewer_rep_writer.write(
                    {"rgb": rp_rgb.get_data(), "trigger_outputs": {"on_time": 0}}
                )
                # rep_writer.write(resize_dict(camera.data.output))
                # wrist_rep_writer.write(resize_dict(wrist_camera.data.output))

            elapsed = time.time() - last_time
            fps = 1.0 / elapsed
            fps_list.append(fps)
            if obs_idx % 10 == 0 and i == 0:
                fps_avg = np.mean(fps_list[-10:])
                print(f"Frames per second: {fps_avg}, (index {obs_idx})")
            last_time = time.time()

            robot.set_dof_state(
                torch.concat(
                    [
                        interp_goal,
                        torch.from_numpy(
                            interp_fj1.reshape(
                                1,
                            )
                        ).float(),
                        torch.from_numpy(
                            interp_fj2.reshape(
                                1,
                            )
                        ).float(),
                    ]
                ),
                torch.concat(
                    [
                        torch.tensor(obs["robot_state"]["joint_velocities"]),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                    ]
                ),
            )

            for j in range(args_cli.num_parts):
                # pos = obs['parts_poses'][7 * i:7 * i + 3]
                goal_pos = data["observations"][part_idx]["parts_poses"][
                    7 * j : 7 * j + 3
                ]
                part_dx = (goal_pos - parts_prev_goal_pos[j]) / sim_steps
                pos = torch.tensor(parts_prev_goal_pos[j] + (i + 1) * part_dx)

                goal_ori = data["observations"][part_idx]["parts_poses"][
                    7 * j + 3 : 7 * j + 7
                ]
                interp_fraction = i / sim_steps
                ori = T.quat_slerp(
                    parts_prev_goal_ori[j], goal_ori, fraction=interp_fraction
                )

                rot = T.quat2mat(ori)
                pose = april_to_sim_mat @ (T.to_homogeneous(pos, rot))
                pos, ori = T.mat2pose(pose)
                ori = T.convert_quat(ori, to="wxyz")
                pos = torch.from_numpy(pos).unsqueeze(0)
                ori = torch.from_numpy(ori).unsqueeze(0)
                views[j].set_world_poses(positions=pos, orientations=ori)

            if (
                base_action_data is not None
                and args_cli.visualize_actions
                and obs_idx < len(base_action_data)
            ):
                # see if we should visualize

                ee_pos_dx = (ee_pos - prev_ee_pos) / sim_steps
                ee_pos_viz = prev_ee_pos + (i + 1) * ee_pos_dx
                interp_fraction = i / sim_steps
                ee_quat_viz = T.quat_slerp(
                    prev_ee_quat, ee_quat, fraction=interp_fraction
                )
                base_action_dx = (base_action_pos - prev_base_action_pos) / sim_steps
                base_action_pos_viz = prev_base_action_pos + (i + 1) * base_action_dx
                full_action_dx = (full_action_pos - prev_full_action_pos) / sim_steps
                full_action_pos_viz = prev_full_action_pos + (i + 1) * full_action_dx
                # base_action_pos = base_action_data[obs_idx]
                # full_action_pos = (
                #     obs["robot_state"]["ee_pos"] + data["actions"][obs_idx][:3]
                # )
                # draw.clear_lines()
                # draw_res_action(
                #     ee_pos=obs["robot_state"]["ee_pos"],
                #     ee_quat=obs["robot_state"]["ee_quat"],
                #     base_action_pos=base_action_pos,
                #     full_action_pos=full_action_pos,
                # )
                draw.clear_lines()
                draw_res_action(
                    ee_pos=ee_pos_viz,
                    ee_quat=ee_quat_viz,
                    base_action_pos=base_action_pos_viz,
                    full_action_pos=full_action_pos_viz,
                )

            # perform step
            sim.step()
            for _ in range(args_cli.sub_steps):
                sim.render()

            # update sim-time
            sim_time += sim_dt
            ep_step_count += 1
            # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
            if sim.is_playing():
                # update buffers
                robot.update_buffers(sim_dt)

            if (
                obs_idx > pause_idx
                and args_cli.visualize_actions
                and run_return_flag < 2
            ):
                run_return_flag = run_until_quit(
                    simulation_app=simulation_app, world=sim
                )

        prev_goal_pos = torch.tensor(obs["robot_state"]["joint_positions"])
        # fj1 = obs["robot_state"]["gripper_finger_1_pos"]
        # fj2 = obs["robot_state"]["gripper_finger_2_pos"]
        prev_fj1 = obs["robot_state"]["gripper_finger_1_pos"]
        prev_fj2 = obs["robot_state"]["gripper_finger_2_pos"]
        parts_prev_goal_pos = []
        parts_prev_goal_ori = []
        for i in range(args_cli.num_parts):
            pos = data["observations"][part_idx]["parts_poses"][7 * i : 7 * i + 3]
            ori = data["observations"][part_idx]["parts_poses"][7 * i + 3 : 7 * i + 7]
            parts_prev_goal_pos.append(pos)
            parts_prev_goal_ori.append(ori)

        if (
            base_action_data is not None
            and args_cli.visualize_actions
            and obs_idx < len(base_action_data)
        ):
            # see if we should visualize
            prev_base_action_pos = base_action_pos
            prev_full_action_pos = full_action_pos
            prev_ee_pos = obs["robot_state"]["ee_pos"]
            prev_ee_quat = obs["robot_state"]["ee_quat"]

        if args_cli.save:
            # log the new observation
            # wrist_image = resize(wrist_camera.data.output["rgb"][:, :, :3].copy())
            # front_image = resize(camera.data.output["rgb"][:, :, :3].copy())

            wrist_image = resize(wrist_camera.data.output["rgb"][:, :, :3])
            front_image = resize(camera.data.output["rgb"][:, :, :3])

            # wrist_image = wrist_camera.data.output["rgb"][:, :, :3]
            # front_image = camera.data.output["rgb"][:, :, :3]
            new_obs = dict(
                color_image1=wrist_image,
                color_image2=front_image,
                robot_state=obs["robot_state"],
            )
            episode_data["observations"].append(new_obs)

        if obs_idx > pause_idx and args_cli.visualize_actions and run_return_flag < 2:
            # st = time.time()
            # while time.time() - st < 0.5:
            #     sim.render()
            run_return_flag = run_until_quit(simulation_app=simulation_app, world=sim)
        else:
            run_return_flag = 1

        if run_return_flag == -1:
            obs_idx -= 1
        else:
            obs_idx += 1

    e = time.time()
    print(f"Time taken: {e-s}")

    if args_cli.save:
        print(f"Saving new pickle to: {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(episode_data, f)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
