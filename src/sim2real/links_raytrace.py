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
parser.add_argument("--headless",
                    action="store_true",
                    default=False,
                    help="Force display off at all times.")
parser.add_argument('-i', '--demo-index',
                    type=int,
                    default=0)
args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": args_cli.headless})
"""Rest everything follows."""

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
from omni.isaac.orbit.utils.math import convert_quat

DEMO_DIR = "/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/low/success"
FILES = [os.path.join(DEMO_DIR, f) for f in os.listdir(DEMO_DIR)]
FILE = FILES[args_cli.demo_index]
FILE = "/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/diffik/sim/one_leg/teleop/low/success/2024-05-09T15:32:54.pkl"

import numpy as np

import furniture_bench.utils.transform as T
from furniture_bench.config import config


april_to_sim_mat = np.array([[6.1232343e-17, 1.0000000e+00, 1.2246469e-16, 1.4999807e-03],
                             [1.0000000e+00, -6.1232343e-17, -7.4987988e-33, 0.0000000e+00],
                             [0.0000000e+00, 1.2246469e-16, -1.0000000e+00, 4.1500002e-01],
                             [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])


def gen_prim(usd_path: str, prim_name: str, init_pos, init_ori):
    # furniture_assets_root_path = Path(f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd")
    furniture_assets_root_path = Path(f"assets/furniture/mesh/usd")
    usd_path = str(furniture_assets_root_path / usd_path)
    pose = april_to_sim_mat @ (T.to_homogeneous(init_pos, init_ori))
    pos, ori = T.mat2pose(pose)
    ori = T.convert_quat(ori, to='wxyz')

    prim_utils.create_prim(prim_name, usd_path=usd_path, translation=pos, orientation=ori)

    view = XFormPrimView(prim_name, reset_xform_properties=False)
    # view = RigidPrimView(prim_name, reset_xform_properties=False)
    return view


def main():
    s = time.time()
    with open(FILE, 'rb') as f:
        data = pickle.load(f)
    """Spawns a single arm manipulator and applies random joint commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera

    # Setup camera sensor
    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=720,
        width=1280,
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg, device='cpu')

    # Spawn camera
    camera.spawn("/World/CameraSensor")

    # Create replicator writer
    demo_date = FILE.split('/')[-1].replace('.pkl', '')
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera", f"{demo_date}_no_april")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    wrist_output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "wrist_camera", f"{demo_date}_no_april")
    wrist_rep_writer = rep.BasicWriter(output_dir=wrist_output_dir, frame_padding=3)

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 5.0),
        attributes={
            "radius": 4.5,
            "intensity": 1200.0,
            "color": (0.75, 0.75, 0.75)
        },
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 5.0),
        attributes={
            "radius": 4.5,
            "intensity": 1200.0,
            "color": (1.0, 1.0, 1.0)
        },
    )
    # Doom linght
    # prim_utils.create_prim(
    #     "/World/Light/DoomLight",
    #     "DoomLight",
    #     translation=(-4.5, 3.5, 10.0),
    #     attributes={
    #         "radius": 4.5,
    #         "intensity": 300.0,
    #         "color": (1.0, 1.0, 1.0)
    #     },
    # )

    # Table
    # table_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

    # usd_base_path = Path(f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd")
    usd_base_path = Path(f"assets/furniture/mesh/usd")
    table_path = (usd_base_path / "table.usda")
    print(f'Table path: {table_path}')
    table_pos = (0., 0., 0.4)
    prim_utils.create_prim(
        "/World/Table",
        usd_path=str(usd_base_path / "table.usda"))
        # translation=table_pos)

    views = []

    # Background
    prim_utils.create_prim(
        '/World/Background',
        translation=(-0.8, 0, 0.75),
        usd_path=str((usd_base_path / "background.usda")))

    # # Base tag
    # prim_utils.create_prim(
    #     '/World/BaseTag',
    #     translation=(0, 0, 0.415),
    #     usd_path=f'{os.path.dirname(os.path.abspath(__file__))}/../assets/furniture/mesh/base_tag.usd')

    # Obstacle front.
    prim_utils.create_prim(
        '/World/ObstacleFront',
        translation=(0.3815, 0., 0.43),
        orientation=(0.7071067690849304, 0, 0, 0.7071067690849304),
        usd_path=str((usd_base_path / "obstacle_front.usda")))

    # Obstacle side.
    prim_utils.create_prim(
        '/World/ObstacleSide1',
        translation=(0.306, -0.175, 0.43),
        orientation=(0.7071067690849304, 0, 0, 0.7071067690849304),
        usd_path=str((usd_base_path / "obstacle_side.usda")))
    prim_utils.create_prim(
        '/World/ObstacleSide2',
        translation=(0.306, 0.175, 0.43),
        orientation=(0.7071067690849304, 0, 0, 0.7071067690849304),
        usd_path=str((usd_base_path / "obstacle_side.usda")))

    # bg_rot = rot_mat([0, 0, np.pi / 2])
    # bg_view = gen_prim('background.usd', , torch.tensor([0.8, 0, 0.75]), bg_rot)

    square_table_top_pos = config['furniture']['square_table']['square_table_top']['reset_pos'][0]
    square_table_top_ori = config['furniture']['square_table']['square_table_top']['reset_ori'][
        0][:3, :3]
    view = gen_prim('square_table/square_table_top_no_tag.usda', '/World/SquareTableTop',
                    square_table_top_pos, square_table_top_ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg1']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg1']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg1_no_tag.usda', '/World/SquareTableLeg1', pos, ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg2']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg2']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg2_no_tag.usda', '/World/SquareTableLeg2', pos, ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg3']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg3']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg3_no_tag.usda', '/World/SquareTableLeg3', pos, ori)
    views.append(view)

    pos = config['furniture']['square_table']['square_table_leg4']['reset_pos'][0]
    ori = config['furniture']['square_table']['square_table_leg4']['reset_ori'][0][:3, :3]
    view = gen_prim('square_table/square_table_leg4_no_tag.usda', '/World/SquareTableLeg4', pos, ori)
    views.append(view)

    # Robots
    # -- Spawn robot
    franka_from_origin_mat = np.array([[1., 0., 0., -0.3], [0., 1., 0., 0.], [0., 0., 1., 0.43],
                                       [0., 0., 0., 1.]])
    robot_pos, robot_ori = T.mat2pose(franka_from_origin_mat)
    # robot_ori = convert_quat(robot_ori, to="wxyz")
    robot_ori = T.convert_quat(robot_ori, to="wxyz")

    prim_name = "/World/Robot_2"
    # asset_path = Path(f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets")
    asset_path = Path(f"sim2real/assets")
    # robot_usd_path = str(asset_path / "franka/franka_usd_instance/franka_instanceable.usd")
    robot_usd_path = str(asset_path / "franka/franka_usd_instance/franka_instanceable_no_joints.usda")
    prim_utils.create_prim(prim_name, usd_path=robot_usd_path, translation=robot_pos + np.array([0., 0., 0.015]), orientation=robot_ori)
    robot_view = XFormPrimView(prim_name, reset_xform_properties=False)

    # Setup camera sensor
    wrist_camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=640,
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    wrist_camera = Camera(cfg=wrist_camera_cfg, device='cpu')

    # Spawn camera
    wrist_camera.spawn("/World/Robot_2/panda_hand",
                       translation=np.array([0.1, 0.0, 0.04]),
                       orientation=T.convert_quat(R.from_euler("XYZ", np.deg2rad([180, 45, -90])).as_quat(), to='wxyz'))

    # Play the simulator
    sim.reset()
    # Acquire handles

    # Initialize camera
    camera.initialize()

    cam_pos = (1.3, -0.00, 0.80)
    cam_target = (-1, -0.00, 0.4)
    camera.set_world_pose_from_view(cam_pos, cam_target)

    wrist_camera.initialize()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics

    # sim_steps = 1.0 / 10 / sim_dt  # #steps / 10Hz / dt
    sim_steps = 1
    print(f'Sim steps: {sim_steps}')

    # # Initialize robot state
    # robot.set_dof_state(
    #     torch.concat([
    #         torch.tensor(data['observations'][0]['robot_state']['joint_positions']),
    #         torch.from_numpy(np.array([data['observations'][0]['robot_state']['gripper_width']])) /
    #         2,
    #         torch.from_numpy(np.array([data['observations'][0]['robot_state']['gripper_width']])) /
    #         2
    #     ]),
    #     torch.concat([
    #         torch.tensor(data['observations'][0]['robot_state']['joint_velocities']),
    #         torch.zeros((1, )),
    #         torch.zeros((1, ))
    #     ]))

    prev_goal_pos = torch.tensor(data['observations'][0]['robot_state']['joint_positions'])

    parts_prev_goal_pos = []
    parts_prev_goal_ori = []
    part_idx_offset = 1

    prev_gripper_width = data['observations'][0]['robot_state']['gripper_width']
    fj1 = data['observations'][0]['robot_state']['finger_joint_1']
    fj2 = data['observations'][0]['robot_state']['finger_joint_2']

    for i in range(5):
        pos = data['observations'][part_idx_offset]['parts_poses'][7 * i:7 * i + 3]
        ori = data['observations'][part_idx_offset]['parts_poses'][7 * i + 3:7 * i + 7]
        parts_prev_goal_pos.append(pos)
        parts_prev_goal_ori.append(ori)

    # for _ in range(100):
    #     sim.step()
    #     time.sleep(0.01)

    # from omni.isaac.core.utils.extensions import get_extension_path_from_name
    # from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    # from omni.isaac.motion_generation import interface_config_loader
    # mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
    # kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
    # kinematics_solver = LulaKinematicsSolver(
    #     robot_description_path = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
    #     urdf_path = kinematics_config_dir + "/franka/lula_franka_gen.urdf"
    # )
    # print("Valid frame names at which to compute kinematics:", kinematics_solver.get_all_frame_names())

    # body_names = robot.articulations.body_names

    links_view = XFormPrimView("/World/Robot_2/panda_link*", name="links_xform_prim_view")
    hand_view = XFormPrimView("/World/Robot_2/panda_hand", name="hand_xform_prim_view")
    right_finger_view = XFormPrimView("/World/Robot_2/panda_rightfinger*", name="left_finger_xform_prim_view")
    left_finger_view = XFormPrimView("/World/Robot_2/panda_leftfinger*", name="right_finger_xform_prim_view")

    from util import create_panda_urdf, compute_world_poses

    # all joints (9?)
    urdf = create_panda_urdf()
    # joint_pos = np.array([-0.1363, -0.0406, -0.0460, -2.1322, 0.0191, 2.0759, 0.5, 0.0])
    # urdf.update_cfg(joint_pos)

    # joint_pos = np.array([-0.1363, -0.0406, -0.0460, -2.1322, 0.0191, 2.0759, 0.5, 0.0, 0.0])
    # world_poses = compute_world_poses(urdf, joint_pos)

    fps_list = []
    last_time = time.time()
    for obs_idx, obs in enumerate(data['observations']):
        # while True:
        goal_pos = torch.tensor(obs['robot_state']['joint_positions'])
        dx = (goal_pos - prev_goal_pos) / sim_steps

        part_idx = obs_idx + part_idx_offset if obs_idx + part_idx_offset < len(
            data['observations']) else len(data['observations']) - 1

        for i in range(int(sim_steps)):
            interp_goal = prev_goal_pos + (i + 1) * dx

            # # If simulation is paused, then skip.
            # if not sim.is_playing():
            #     print(f'Skipping...')
            #     sim.step(render=not args_cli.headless)
            #     continue

            # Update camera data
            camera.update(dt=0.0)
            wrist_camera.update(dt=0.0)

            # from IPython import embed; embed()
            # assert False
            rep_writer.write(convert_dict_to_backend(camera.data.output, backend="numpy"))
            wrist_rep_writer.write(convert_dict_to_backend(wrist_camera.data.output, backend="numpy"))

            # elapsed = time.time() - last_time
            # fps = 1.0 / elapsed
            # fps_list.append(fps)
            # if i % 10 == 0:
            #     fps_avg = np.mean(fps_list[-10:])
            #     print(f'Frames per second: {fps_avg}')
            # last_time = time.time()

            griper_dx = (obs['robot_state']['gripper_width'] - prev_gripper_width) / sim_steps
            gripper_interp_goal = prev_gripper_width + (i + 1) * griper_dx

            # robot.set_dof_state(
            #     torch.concat([
            #         interp_goal,
            #         torch.from_numpy(np.array([gripper_interp_goal])).float() / 2,
            #         torch.from_numpy(np.array([gripper_interp_goal])).float() / 2
            #     ]),
            #     torch.concat([
            #         torch.tensor(obs['robot_state']['joint_velocities']),
            #         torch.zeros((1, )),
            #         torch.zeros((1, ))
            #     ]))

            # from IPython import embed; embed()
            # assert False

            full_dof_state = np.concatenate([
                interp_goal.cpu().numpy(),
                np.array([0.0]),
                # np.array([gripper_interp_goal]) / 2,
                # np.array([gripper_interp_goal]) / 2
                fj1.reshape(1,),
                fj2.reshape(1,)
            ], axis=-1)
            world_poses_mat_list, tf_pcd_list = compute_world_poses(urdf, full_dof_state)
            world_poses_mat = np.stack(world_poses_mat_list)
            world_pos_arr = world_poses_mat[:, :-1, -1] + robot_pos
            # world_ori_arr = convert_quat(R.from_matrix(world_poses_mat[:, :-1, :-1]).as_quat(), to="wxyz")
            world_ori_arr = np.vstack([T.convert_quat(R.from_matrix(world_poses_mat[idx, :-1, :-1]).as_quat(), to="wxyz") for idx in range(world_poses_mat.shape[0])])

            link_pos_tensor = torch.from_numpy(world_pos_arr)[:7]
            link_ori_tensor = torch.from_numpy(world_ori_arr)[:7]

            hand_pos_tensor = torch.from_numpy(world_pos_arr)[7].reshape(1, 3)
            hand_ori_tensor = torch.from_numpy(world_ori_arr)[7].reshape(1, 4)

            lfinger_pos_tensor = torch.from_numpy(world_pos_arr)[8].reshape(1, 3)
            lfinger_ori_tensor = torch.from_numpy(world_ori_arr)[8].reshape(1, 4)

            rfinger_pos_tensor = torch.from_numpy(world_pos_arr)[9].reshape(1, 3)
            rfinger_ori_tensor = torch.from_numpy(world_ori_arr)[9].reshape(1, 4)

            links_view.set_world_poses(positions=torch.Tensor([[-0.3, 0.0, 0.43 + 0.015]]), orientations=torch.Tensor([[1.0, 0.0, 0.0, 0.0]]), indices=torch.Tensor([0]))
            links_view.set_world_poses(positions=link_pos_tensor, orientations=link_ori_tensor, indices=torch.arange(1, 8))
            hand_view.set_world_poses(positions=hand_pos_tensor, orientations=hand_ori_tensor)
            right_finger_view.set_world_poses(positions=rfinger_pos_tensor, orientations=rfinger_ori_tensor)
            left_finger_view.set_world_poses(positions=lfinger_pos_tensor, orientations=lfinger_ori_tensor)

            for j in range(5):
                # pos = obs['parts_poses'][7 * i:7 * i + 3]
                goal_pos = data['observations'][part_idx]['parts_poses'][7 * j:7 * j + 3]
                part_dx = (goal_pos - parts_prev_goal_pos[j]) / sim_steps
                pos = torch.tensor(parts_prev_goal_pos[j] + (i + 1) * part_dx)

                goal_ori = data['observations'][part_idx]['parts_poses'][7 * j + 3:7 * j + 7]
                interp_fraction = i / sim_steps
                ori = T.quat_slerp(parts_prev_goal_ori[j], goal_ori, fraction=interp_fraction)

                rot = T.quat2mat(ori)
                pose = april_to_sim_mat @ (T.to_homogeneous(pos, rot))
                pos, ori = T.mat2pose(pose)
                ori = T.convert_quat(ori, to='wxyz')
                pos = torch.from_numpy(pos).unsqueeze(0)
                ori = torch.from_numpy(ori).unsqueeze(0)
                views[j].set_world_poses(positions=pos, orientations=ori)

            # perform step
            # sim.step()
            sim.render()
            
            # update sim-time
            sim_time += sim_dt
            ep_step_count += 1

            # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
            # if sim.is_playing():
            #     # update buffers
            #     robot.update_buffers(sim_dt)


        prev_goal_pos = torch.tensor(obs['robot_state']['joint_positions'])
        prev_gripper_width = obs['robot_state']['gripper_width']
        fj1 = obs['robot_state']['finger_joint_1']
        fj2 = obs['robot_state']['finger_joint_2']
        parts_prev_goal_pos = []
        parts_prev_goal_ori = []
        for i in range(5):
            pos = data['observations'][part_idx]['parts_poses'][7 * i:7 * i + 3]
            ori = data['observations'][part_idx]['parts_poses'][7 * i + 3:7 * i + 7]
            parts_prev_goal_pos.append(pos)
            parts_prev_goal_ori.append(ori)

    e=time.time()
    print(f"Time taken: {e-s}")

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
