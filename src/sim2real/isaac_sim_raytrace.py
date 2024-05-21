# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import argparse
import time
import os
from pathlib import Path
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
args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp(
    {
        "headless": args_cli.headless,
        "width": 640,
        "height": 480,
    }
)
"""Rest everything follows."""

import numpy as np
import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
import omni.replicator.core as rep
from omni.isaac.orbit.utils import convert_dict_to_backend

import furniture_bench.utils.transform as T
from furniture_bench.config import config

from typing import Dict

# folder to load from
# DEMO_DIR = "/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/med_perturb/success"
DEMO_DIR = args_cli.load_dir
FILES = [os.path.join(DEMO_DIR, f) for f in os.listdir(DEMO_DIR)]
FILE = FILES[args_cli.demo_index]

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

    global RP_RUNNING, RP_PAUSE
    RP_RUNNING = True
    RP_PAUSE = False

    def on_kb_input(e):
        global RP_RUNNING, RP_PAUSE
        if e.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if e.input == carb.input.KeyboardInput.Q:
                print(f'[Run until quit] Caught "Q" keypress, closing app')
                RP_RUNNING = False
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

    return


def gen_prim(usd_path: str, prim_name: str, init_pos, init_ori):
    furniture_assets_root_path = Path(
        f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd"
    )
    # furniture_assets_root_path = Path(f"assets/furniture/mesh/usd")
    usd_path = str(furniture_assets_root_path / usd_path)
    pose = april_to_sim_mat @ (T.to_homogeneous(init_pos, init_ori))
    pos, ori = T.mat2pose(pose)
    ori = T.convert_quat(ori, to="wxyz")

    prim_utils.create_prim(
        prim_name, usd_path=usd_path, translation=pos, orientation=ori
    )

    view = XFormPrimView(prim_name, reset_xform_properties=False)
    return view


def main():
    s = time.time()
    with open(FILE, "rb") as f:
        data = pickle.load(f)
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

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 5.0),
        attributes={"radius": 4.5, "intensity": 1200.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 5.0),
        attributes={"radius": 4.5, "intensity": 1200.0, "color": (1.0, 1.0, 1.0)},
    )

    # Table
    usd_base_path = Path(
        f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd"
    )
    # usd_base_path = Path(f"assets/furniture/mesh/usd")
    table_z_offset = 0.415

    prim_utils.create_prim("/World/Table", usd_path=str(usd_base_path / "table.usda"))
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
    square_table_top_pos = config["furniture"]["square_table"]["square_table_top"][
        "reset_pos"
    ][0]
    square_table_top_ori = config["furniture"]["square_table"]["square_table_top"][
        "reset_ori"
    ][0][:3, :3]
    view = gen_prim(
        "square_table/square_table_top_no_tag.usda",
        "/World/SquareTableTop",
        square_table_top_pos,
        square_table_top_ori,
    )
    views.append(view)

    pos = config["furniture"]["square_table"]["square_table_leg1"]["reset_pos"][0]
    ori = config["furniture"]["square_table"]["square_table_leg1"]["reset_ori"][0][
        :3, :3
    ]
    view = gen_prim(
        "square_table/square_table_leg1_no_tag.usda", "/World/SquareTableLeg1", pos, ori
    )
    views.append(view)

    pos = config["furniture"]["square_table"]["square_table_leg2"]["reset_pos"][0]
    ori = config["furniture"]["square_table"]["square_table_leg2"]["reset_ori"][0][
        :3, :3
    ]
    view = gen_prim(
        "square_table/square_table_leg2_no_tag.usda", "/World/SquareTableLeg2", pos, ori
    )
    views.append(view)

    pos = config["furniture"]["square_table"]["square_table_leg3"]["reset_pos"][0]
    ori = config["furniture"]["square_table"]["square_table_leg3"]["reset_ori"][0][
        :3, :3
    ]
    view = gen_prim(
        "square_table/square_table_leg3_no_tag.usda", "/World/SquareTableLeg3", pos, ori
    )
    views.append(view)

    pos = config["furniture"]["square_table"]["square_table_leg4"]["reset_pos"][0]
    ori = config["furniture"]["square_table"]["square_table_leg4"]["reset_ori"][0][
        :3, :3
    ]
    view = gen_prim(
        "square_table/square_table_leg4_no_tag.usda", "/World/SquareTableLeg4", pos, ori
    )
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
        f"{demo_date}_substeps_{args_cli.sub_steps}",
    )
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    wrist_output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "output",
        "wrist_camera",
        f"{demo_date}_substeps_{args_cli.sub_steps}",
    )
    wrist_rep_writer = rep.BasicWriter(output_dir=wrist_output_dir, frame_padding=3)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    ep_step_count = 0

    sim_steps = 1
    print(f"Sim steps: {sim_steps}")

    # Initialize robot state
    robot.set_dof_state(
        torch.concat(
            [
                torch.tensor(data["observations"][0]["robot_state"]["joint_positions"]),
                torch.from_numpy(
                    np.array([data["observations"][0]["robot_state"]["gripper_width"]])
                )
                / 2,
                torch.from_numpy(
                    np.array([data["observations"][0]["robot_state"]["gripper_width"]])
                )
                / 2,
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

    prev_goal_pos = torch.tensor(
        data["observations"][0]["robot_state"]["joint_positions"]
    )

    parts_prev_goal_pos = []
    parts_prev_goal_ori = []
    part_idx_offset = 1

    fj1 = data["observations"][0]["robot_state"]["gripper_finger_1_pos"]
    fj2 = data["observations"][0]["robot_state"]["gripper_finger_2_pos"]

    # initialize parts
    for i in range(5):
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

    # run_until_quit(simulation_app=simulation_app, world=sim)
    # from IPython import embed; embed()
    # assert False

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
    episode_data["metadata"] = data["metadata"]

    fps_list = []
    last_time = time.time()
    for obs_idx, obs in enumerate(data["observations"]):
        goal_pos = torch.tensor(obs["robot_state"]["joint_positions"])
        dx = (goal_pos - prev_goal_pos) / sim_steps

        part_idx = (
            obs_idx + part_idx_offset
            if obs_idx + part_idx_offset < len(data["observations"])
            else len(data["observations"]) - 1
        )

        # if obs_idx == 50:
        #     run_until_quit(simulation_app=simulation_app, world=sim)
        #     from IPython import embed; embed()
        #     assert False

        for i in range(int(sim_steps)):
            interp_goal = prev_goal_pos + (i + 1) * dx

            # If simulation is paused, then skip.
            if not sim.is_playing():
                sim.step(render=not args_cli.headless)
                continue

            # Update camera data
            camera.update(dt=0.0)
            wrist_camera.update(dt=0.0)

            rep_writer.write(
                convert_dict_to_backend(camera.data.output, backend="numpy")
            )
            wrist_rep_writer.write(
                convert_dict_to_backend(wrist_camera.data.output, backend="numpy")
            )
            elapsed = time.time() - last_time
            fps = 1.0 / elapsed
            fps_list.append(fps)
            if i % 10 == 0:
                fps_avg = np.mean(fps_list[-10:])
                print(f"Frames per second: {fps_avg}")
            last_time = time.time()

            robot.set_dof_state(
                torch.concat(
                    [
                        interp_goal,
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
                        torch.tensor(obs["robot_state"]["joint_velocities"]),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                    ]
                ),
            )

            for j in range(5):
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

        prev_goal_pos = torch.tensor(obs["robot_state"]["joint_positions"])
        fj1 = obs["robot_state"]["gripper_finger_1_pos"]
        fj2 = obs["robot_state"]["gripper_finger_2_pos"]
        parts_prev_goal_pos = []
        parts_prev_goal_ori = []
        for i in range(5):
            pos = data["observations"][part_idx]["parts_poses"][7 * i : 7 * i + 3]
            ori = data["observations"][part_idx]["parts_poses"][7 * i + 3 : 7 * i + 7]
            parts_prev_goal_pos.append(pos)
            parts_prev_goal_ori.append(ori)

        # log the new observation
        wrist_image = wrist_camera.data.output["rgb"][:, :, :3]
        front_image = camera.data.output["rgb"][:, :, :3]
        new_obs = dict(
            color_image1=wrist_image,
            color_image2=front_image,
            robot_state=obs["robot_state"],
        )
        episode_data["observations"].append(new_obs)

    e = time.time()
    print(f"Time taken: {e-s}")

    print(f"Saving new pickle to: {pkl_path}")
    with open(pkl_path, "wb") as f:
        pickle.dump(episode_data, f)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
