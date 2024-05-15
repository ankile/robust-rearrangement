try:
    import isaacgym
    from isaacgym import gymapi, gymtorch
except ImportError as e:
    from rich import print

    print(
        """[red][Isaac Gym Import Error]
  1. You need to install Isaac Gym, if not installed.
    - Download Isaac Gym following https://clvrai.github.io/furniture-bench/docs/getting_started/installation_guide_furniture_sim.html#download-isaac-gym
    - Then, pip install -e isaacgym/python
  2. If PyTorch was imported before furniture_bench, please import torch after furniture_bench.[/red]
"""
    )
    print()
    raise ImportError(e)


from typing import Union
from datetime import datetime
from pathlib import Path

from furniture_bench.furniture.furniture import Furniture
from src.dataset.normalizer import LinearNormalizer
import torch
import cv2
import gym
import numpy as np

import pytorch3d.transforms as pt

import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.envs.initialization_mode import Randomness, str_to_enum
from src.controllers.diffik import diffik_factory

# from furniture_bench.controllers.diffik_qp import diffik_factory
from furniture_bench.furniture import furniture_factory
from furniture_bench.sim_config import sim_config
from furniture_bench.config import ROBOT_HEIGHT, config
from furniture_bench.utils.pose import get_mat, rot_mat
from furniture_bench.envs.observation import (
    DEFAULT_VISUAL_OBS,
)
from furniture_bench.robot.robot_state import ROBOT_STATE_DIMS
from furniture_bench.furniture.parts.part import Part
from src.common.geometry import proprioceptive_quat_to_6d_rotation

from ipdb import set_trace as bp


ASSET_ROOT = str(Path(__file__).parent.parent.absolute() / "assets")


class FurnitureSimEnv(gym.Env):
    """FurnitureSim base class."""

    def __init__(
        self,
        furniture: str,
        num_envs: int = 1,
        resize_img: bool = True,
        obs_keys=None,
        concat_robot_state: bool = False,
        manual_label: bool = False,
        manual_done: bool = False,
        headless: bool = False,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        init_assembled: bool = False,
        np_step_out: bool = False,
        channel_first: bool = False,
        randomness: Union[str, Randomness] = "low",
        high_random_idx: int = 0,
        save_camera_input: bool = False,
        record: bool = False,
        max_env_steps: int = 3000,
        act_rot_repr: str = "rot_6d",
        action_type: str = "delta",  # "delta" or "pos"
        ctrl_mode: str = "diffik",
        ee_laser: bool = False,
        **kwargs,
    ):
        """
        Args:
            furniture (str): Specifies the type of furniture. Options are 'lamp', 'square_table', 'desk', 'drawer', 'cabinet', 'round_table', 'stool', 'chair', 'one_leg'.
            num_envs (int): Number of parallel environments.
            resize_img (bool): If true, images are resized to 224 x 224.
            obs_keys (list): List of observations for observation space (i.e., RGB-D image from three cameras, proprioceptive states, and poses of the furniture parts.)
            concat_robot_state (bool): Whether to return concatenated `robot_state` or its dictionary form in observation.
            manual_label (bool): If true, the environment reward is manually labeled.
            manual_done (bool): If true, the environment is terminated manually.
            headless (bool): If true, simulation runs without GUI.
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            init_assembled (bool): If true, the environment is initialized with assembled furniture.
            np_step_out (bool): If true, env.step() returns Numpy arrays.
            channel_first (bool): If true, color images are returned in channel first format [3, H, w].
            randomness (str): Level of randomness in the environment. Options are 'low', 'med', 'high'.
            high_random_idx (int): Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
            save_camera_input (bool): If true, the initial camera inputs are saved.
            record (bool): If true, videos of the wrist and front cameras' RGB inputs are recorded.
            max_env_steps (int): Maximum number of steps per episode (default: 3000).
            act_rot_repr (str): Representation of rotation for action space. Options are 'quat', 'axis', or 'rot_6d'.
            ctrl_mode (str): 'diffik' (joint impedance, with differential inverse kinematics control)
        """
        super(FurnitureSimEnv, self).__init__()
        self.device = torch.device("cuda", compute_device_id)

        assert (
            ctrl_mode == "diffik"
        ), "Only 'diffik' controller is supported for now (parallization)."

        # NOTE: This is solely for the grid search
        self.pos_scalar = kwargs.get("pos_scalar", 1.0)
        self.rot_scalar = kwargs.get("rot_scalar", 1.0)
        self.stiffness = kwargs.get("stiffness", 1000.0)
        self.damping = kwargs.get("damping", 200.0)

        print(
            f"Making DiffIK controller with pos_scalar: {self.pos_scalar}, rot_scalar: {self.rot_scalar}"
        )
        print(f"Stiffness: {self.stiffness}, Damping: {self.damping}")

        self.assemble_idx = 0
        # Furniture for each environment (reward, reset).
        self.furnitures = [furniture_factory(furniture) for _ in range(num_envs)]

        if num_envs == 1:
            self.furniture = self.furnitures[0]
        else:
            self.furniture = furniture_factory(furniture)

        self.furniture.max_env_steps = max_env_steps
        for furn in self.furnitures:
            furn.max_env_steps = max_env_steps

        self.furniture_name = furniture
        self.num_envs = num_envs
        self.obs_keys = obs_keys or DEFAULT_VISUAL_OBS

        print(f"Observation keys: {self.obs_keys}")
        self.robot_state_keys = [
            k.split("/")[1] for k in self.obs_keys if k.startswith("robot_state")
        ]
        self.concat_robot_state = concat_robot_state
        self.pose_dim = 7
        self.resize_img = resize_img
        self.manual_label = manual_label
        self.manual_done = manual_done
        self.headless = headless
        self.render_cameras: bool = any(["image" in k for k in self.obs_keys])

        self.move_neutral = False
        self.ctrl_started = False
        self.init_assembled = init_assembled
        self.np_step_out = np_step_out
        self.channel_first = channel_first
        self.from_skill = (
            0  # TODO: Skill benchmark should be implemented in FurnitureSim.
        )
        self.randomness = str_to_enum(randomness)
        self.high_random_idx = high_random_idx
        self.last_grasp = torch.tensor([-1.0] * num_envs, device=self.device)
        self.grasp_margin = 0.02 - 0.001  # To prevent repeating open and close actions.
        self.max_gripper_width = config["robot"]["max_gripper_width"][furniture]

        self.save_camera_input = save_camera_input
        self.img_size = sim_config["camera"][
            "resized_img_size" if resize_img else "color_img_size"
        ]

        # Simulator setup.
        self.isaac_gym = gymapi.acquire_gym()
        self.sim = self.isaac_gym.create_sim(
            compute_device_id,
            graphics_device_id,
            gymapi.SimType.SIM_PHYSX,
            sim_config["sim_params"],
        )

        # our flags
        self.ctrl_mode = ctrl_mode
        self.ee_laser = ee_laser

        self._create_ground_plane()
        self._setup_lights()
        self.import_assets()
        self.create_envs()
        self.set_viewer()
        self.set_camera()
        self.acquire_base_tensors()

        self.isaac_gym.prepare_sim(self.sim)
        self.refresh()

        self.isaac_gym.refresh_actor_root_state_tensor(self.sim)

        self.init_ee_pos, self.init_ee_quat = self.get_ee_pose()
        self.init_ctrl()

        gym.logger.set_level(gym.logger.INFO)

        self.record = record
        if self.record:
            record_dir = Path("sim_record") / datetime.now().strftime("%Y%m%d-%H%M%S")
            record_dir.mkdir(parents=True, exist_ok=True)
            self.video_writer = cv2.VideoWriter(
                str(record_dir / "video.mp4"),
                cv2.VideoWriter_fourcc(*"MP4V"),
                30,
                (self.img_size[1] * 2, self.img_size[0]),  # Wrist and front cameras.
            )

        if (
            act_rot_repr != "quat"
            and act_rot_repr != "axis"
            and act_rot_repr != "rot_6d"
        ):
            raise ValueError(f"Invalid rotation representation: {act_rot_repr}")
        self.act_rot_repr = act_rot_repr
        self.action_type = action_type

        # Create the action space limits on device here to save computation.
        self.act_low = torch.from_numpy(self.action_space.low).to(device=self.device)
        self.act_high = torch.from_numpy(self.action_space.high).to(device=self.device)
        self.sim_steps = int(
            1.0 / config["robot"]["hz"] / sim_config["sim_params"].dt + 0.1
        )

        print(f"Sim steps: {self.sim_steps}")

        self.robot_state_as_dict = kwargs.get("robot_state_as_dict", True)
        self.squeeze_batch_dim = kwargs.get("squeeze_batch_dim", False)

    def _create_ground_plane(self):
        """Creates ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.isaac_gym.add_ground(self.sim, plane_params)

    def _setup_lights(self):
        for light in sim_config["lights"]:
            l_color = gymapi.Vec3(*light["color"])
            l_ambient = gymapi.Vec3(*light["ambient"])
            l_direction = gymapi.Vec3(*light["direction"])
            self.isaac_gym.set_light_parameters(
                self.sim, 0, l_color, l_ambient, l_direction
            )

    def create_envs(self):
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.franka_pose = gymapi.Transform()

        table_half_width = 0.015
        table_surface_z = table_pos.z + table_half_width
        self.franka_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
        )

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p.x, self.franka_pose.p.y, self.franka_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]

        franka_link_dict = self.isaac_gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_ee_index = franka_link_dict["k_ee_link"]
        self.franka_base_index = franka_link_dict["panda_link0"]
        # Parts assets.
        # Create assets.
        self.part_assets = {}
        for part in self.furniture.parts:
            asset_option = sim_config["asset"][part.name]
            self.part_assets[part.name] = self.isaac_gym.load_asset(
                self.sim, ASSET_ROOT, part.asset_file, asset_option
            )
        # Create envs.
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.env_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.handles = {}
        self.ee_idxs = []
        self.ee_handles = []

        self.base_idxs = []
        self.part_idxs = {}
        self.obstacle_actor_idxs = []
        self.franka_handles = []
        for i in range(self.num_envs):
            env = self.isaac_gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            # Add workspace (table).
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)

            table_handle = self.isaac_gym.create_actor(
                env, self.table_asset, table_pose, "table", i, 0
            )
            table_props = self.isaac_gym.get_actor_rigid_shape_properties(
                env, table_handle
            )
            table_props[0].friction = sim_config["table"]["friction"]
            self.isaac_gym.set_actor_rigid_shape_properties(
                env, table_handle, table_props
            )

            self.base_tag_pose = gymapi.Transform()
            base_tag_pos = T.pos_from_mat(config["robot"]["tag_base_from_robot_base"])
            self.base_tag_pose.p = self.franka_pose.p + gymapi.Vec3(
                base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT
            )
            self.base_tag_pose.p.z = table_surface_z
            base_tag_handle = self.isaac_gym.create_actor(
                env, self.base_tag_asset, self.base_tag_pose, "base_tag", i, 0
            )
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)
            bg_handle = self.isaac_gym.create_actor(
                env, self.background_asset, bg_pose, "background", i, 0
            )
            # TODO: Make config
            obstacle_pose = gymapi.Transform()
            obstacle_pose.p = gymapi.Vec3(
                self.base_tag_pose.p.x + 0.37 + 0.01,
                0.0,
                table_surface_z + 0.015,
            )
            obstacle_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(0, 0, 1), 0.5 * np.pi
            )

            self.obstacle_reset_pose = obstacle_pose

            self.obstacle_handle = self.isaac_gym.create_actor(
                env, self.obstacle_asset, obstacle_pose, f"obstacle", i, 0
            )

            part_idx = self.isaac_gym.get_actor_rigid_body_index(
                env, self.obstacle_handle, 0, gymapi.DOMAIN_SIM
            )
            if self.part_idxs.get("obstacle") is None:
                self.part_idxs["obstacle"] = [part_idx]
            else:
                self.part_idxs[f"obstacle"].append(part_idx)

            # Get the obstacle actor index
            obstacle_actor_idx = self.isaac_gym.find_actor_index(
                env, "obstacle", gymapi.DOMAIN_SIM
            )
            self.obstacle_actor_idxs.append(obstacle_actor_idx)

            # Add robot.
            franka_handle = self.isaac_gym.create_actor(
                env, self.franka_asset, self.franka_pose, "franka", i, 0
            )
            self.franka_num_dofs = self.isaac_gym.get_actor_dof_count(
                env, franka_handle
            )

            self.isaac_gym.enable_actor_dof_force_sensors(env, franka_handle)
            self.franka_handles.append(franka_handle)

            # Get global index of hand and base.
            self.ee_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_ee_index, gymapi.DOMAIN_SIM
                )
            )
            self.ee_handles.append(
                self.isaac_gym.find_actor_rigid_body_handle(
                    env, franka_handle, "k_ee_link"
                )
            )
            self.base_idxs.append(
                self.isaac_gym.get_actor_rigid_body_index(
                    env, franka_handle, self.franka_base_index, gymapi.DOMAIN_SIM
                )
            )
            # Set dof properties.
            franka_dof_props = self.isaac_gym.get_asset_dof_properties(
                self.franka_asset
            )
            if self.ctrl_mode == "osc":
                franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
                franka_dof_props["stiffness"][:7].fill(0.0)
                franka_dof_props["damping"][:7].fill(0.0)
                franka_dof_props["friction"][:7] = sim_config["robot"]["arm_frictions"]
            else:
                franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
                franka_dof_props["stiffness"][:7].fill(self.stiffness)
                franka_dof_props["damping"][:7].fill(self.damping)
                # print(f"Stiffness: {self.stiffness}, Damping: {self.damping}")
            # Grippers
            franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][7:].fill(0)
            franka_dof_props["damping"][7:].fill(0)
            franka_dof_props["friction"][7:] = sim_config["robot"]["gripper_frictions"]
            franka_dof_props["upper"][7:] = self.max_gripper_width / 2

            self.isaac_gym.set_actor_dof_properties(
                env, franka_handle, franka_dof_props
            )
            # Set initial dof states
            franka_num_dofs = self.isaac_gym.get_asset_dof_count(self.franka_asset)
            self.default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
            self.default_dof_pos[:7] = np.array(
                config["robot"]["reset_joints"], dtype=np.float32
            )
            self.default_dof_pos[7:] = self.max_gripper_width / 2
            default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"] = self.default_dof_pos
            self.isaac_gym.set_actor_dof_states(
                env, franka_handle, default_dof_state, gymapi.STATE_ALL
            )
            # Add furniture parts.
            poses = []
            for part in self.furniture.parts:
                pos, ori = self._get_reset_pose(part)
                part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
                part_pose = gymapi.Transform()
                part_pose.p = gymapi.Vec3(
                    part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
                )
                reset_ori = self.april_coord_to_sim_coord(ori)
                part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
                poses.append(part_pose)
                part_handle = self.isaac_gym.create_actor(
                    env, self.part_assets[part.name], part_pose, part.name, i, 0
                )
                self.handles[part.name] = part_handle

                part_idx = self.isaac_gym.get_actor_rigid_body_index(
                    env, part_handle, 0, gymapi.DOMAIN_SIM
                )
                # Set properties of part.
                part_props = self.isaac_gym.get_actor_rigid_shape_properties(
                    env, part_handle
                )
                part_props[0].friction = sim_config["parts"]["friction"]
                self.isaac_gym.set_actor_rigid_shape_properties(
                    env, part_handle, part_props
                )

                if self.part_idxs.get(part.name) is None:
                    self.part_idxs[part.name] = [part_idx]
                else:
                    self.part_idxs[part.name].append(part_idx)

        # This only needs to happen once
        self.parts_handles = {}
        for part in self.furniture.parts:
            self.parts_handles[part.name] = self.isaac_gym.find_actor_index(
                self.envs[0], part.name, gymapi.DOMAIN_ENV
            )

        # print(f'Getting the separate actor indices for the frankas and the furniture parts (not the handles)')
        self.franka_actor_idx_all = []
        self.part_actor_idx_all = []  # global list of indices, when resetting all parts
        self.part_actor_idx_by_env = (
            {}
        )  # allow to access part indices based on environment indices
        for env_idx in range(self.num_envs):
            self.franka_actor_idx_all.append(
                self.isaac_gym.find_actor_index(
                    self.envs[env_idx], "franka", gymapi.DOMAIN_SIM
                )
            )
            self.part_actor_idx_by_env[env_idx] = []
            for part in self.furnitures[env_idx].parts:
                part_actor_idx = self.isaac_gym.find_actor_index(
                    self.envs[env_idx], part.name, gymapi.DOMAIN_SIM
                )
                self.part_actor_idx_all.append(part_actor_idx)
                self.part_actor_idx_by_env[env_idx].append(part_actor_idx)

        self.obstacle_actor_idxs_by_env = torch.tensor(
            self.obstacle_actor_idxs, device=self.device, dtype=torch.int32
        ).unsqueeze(1)

        self.franka_actor_idxs_all_t = torch.tensor(
            self.franka_actor_idx_all, device=self.device, dtype=torch.int32
        )
        self.part_actor_idxs_all_t = torch.tensor(
            self.part_actor_idx_all, device=self.device, dtype=torch.int32
        )

    def _get_reset_pose(self, part: Part):
        """Get the reset pose of the part.

        Args:
            part: The part to get the reset pose.
        """
        if self.init_assembled:
            if part.name == "chair_seat":
                # Special case handling for chair seat since the assembly of chair back is not available from initialized pose.
                part.reset_pos = [[0, 0.16, -0.035]]
                part.reset_ori = [rot_mat([np.pi, 0, 0], hom=True)]
            attached_part = False
            attach_to = None
            for assemble_pair in self.furniture.should_be_assembled:
                if part.part_idx == assemble_pair[1]:
                    attached_part = True
                    attach_to = self.furniture.parts[assemble_pair[0]]
                    break
            if attached_part:
                attach_part_pos = self.furniture.parts[attach_to.part_idx].reset_pos[0]
                attach_part_ori = self.furniture.parts[attach_to.part_idx].reset_ori[0]
                attach_part_pose = get_mat(attach_part_pos, attach_part_ori)
                if part.default_assembled_pose is not None:
                    pose = attach_part_pose @ part.default_assembled_pose
                    pos = pose[:3, 3]
                    ori = T.to_hom_ori(pose[:3, :3])
                else:
                    pos = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0][:4, 3]
                    )
                    pos = pos[:3]
                    ori = (
                        attach_part_pose
                        @ self.furniture.assembled_rel_poses[
                            (attach_to.part_idx, part.part_idx)
                        ][0]
                    )
                part.reset_pos[0] = pos
                part.reset_ori[0] = ori
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        else:
            pos = part.reset_pos[self.from_skill]
            ori = part.reset_ori[self.from_skill]
        return pos, ori

    def set_viewer(self):
        """Create the viewer."""
        self.enable_viewer_sync = True
        self.viewer = None

        if not self.headless:
            self.viewer = self.isaac_gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )
            # Point camera at middle env.
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.62)
            middle_env = self.envs[0]
            self.isaac_gym.viewer_camera_look_at(
                self.viewer, middle_env, cam_pos, cam_target
            )

    def set_camera(self):
        self.camera_handles = {}
        self.camera_obs = {}

        def create_camera(name, i):
            env = self.envs[i]
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = self.img_size[0]
            camera_cfg.height = self.img_size[1]
            camera_cfg.near_plane = 0.001
            camera_cfg.far_plane = 2.0
            camera_cfg.horizontal_fov = 40.0 if self.resize_img else 69.4
            self.camera_cfg = camera_cfg

            if name == "wrist":
                if self.resize_img:
                    camera_cfg.horizontal_fov = 55.0  # Wide view.
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(-0.04, 0, -0.05)
                transform.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0, 1, 0), np.radians(-70.0)
                )
                self.isaac_gym.attach_camera_to_body(
                    camera, env, self.ee_handles[i], transform, gymapi.FOLLOW_TRANSFORM
                )
            elif name == "front":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                cam_pos = gymapi.Vec3(0.90, -0.00, 0.65)
                cam_target = gymapi.Vec3(-1, -0.00, 0.3)
                self.isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
                self.front_cam_pos = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
                self.front_cam_target = np.array(
                    [cam_target.x, cam_target.y, cam_target.z]
                )
            elif name == "rear":
                camera = self.isaac_gym.create_camera_sensor(env, camera_cfg)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(
                    self.franka_pose.p.x + 0.08, 0, self.franka_pose.p.z + 0.2
                )
                transform.r = gymapi.Quat.from_axis_angle(
                    gymapi.Vec3(0, 1, 0), np.radians(35.0)
                )
                self.isaac_gym.set_camera_transform(camera, env, transform)
            return camera

        camera_names = {"1": "wrist", "2": "front"}
        for env_idx, env in enumerate(self.envs):
            for k in self.obs_keys:
                if k.startswith("color"):
                    camera_name = camera_names[k[-1]]
                    render_type = gymapi.IMAGE_COLOR
                elif k.startswith("depth"):
                    camera_name = camera_names[k[-1]]
                    render_type = gymapi.IMAGE_DEPTH
                else:
                    continue
                if camera_name not in self.camera_handles:
                    self.camera_handles[camera_name] = []
                # Only when the camera handle for the current environment does not exist.
                if len(self.camera_handles[camera_name]) <= env_idx:
                    self.camera_handles[camera_name].append(
                        create_camera(camera_name, env_idx)
                    )
                handle = self.camera_handles[camera_name][env_idx]
                tensor = gymtorch.wrap_tensor(
                    self.isaac_gym.get_camera_image_gpu_tensor(
                        self.sim, env, handle, render_type
                    )
                )
                if k not in self.camera_obs:
                    self.camera_obs[k] = []
                self.camera_obs[k].append(tensor)

    def import_assets(self):
        self.base_tag_asset = self._import_base_tag_asset()
        self.background_asset = self._import_background_asset()
        self.table_asset = self._import_table_asset()
        # self.obstacle_front_asset = self._import_obstacle_front_asset()
        # self.obstacle_side_asset = self._import_obstacle_side_asset()
        self.obstacle_asset = self._import_obstacle_asset()
        self.franka_asset = self._import_franka_asset()

    def acquire_base_tensors(self):
        # Get rigid body state tensor
        _rb_states = self.isaac_gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        _root_tensor = self.isaac_gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.root_pos = self.root_tensor.view(self.num_envs, -1, 13)[..., 0:3]
        self.root_quat = self.root_tensor.view(self.num_envs, -1, 13)[..., 3:7]

        _forces = self.isaac_gym.acquire_dof_force_tensor(self.sim)
        _forces = gymtorch.wrap_tensor(_forces)
        self.forces = _forces.view(self.num_envs, 9)

        # Get DoF tensor
        _dof_states = self.isaac_gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(
            _dof_states
        )  # (num_dofs, 2), 2 for pos and vel.
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 9)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 9)
        # Get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.isaac_gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to franka hand
        self.jacobian_eef = self.jacobian[
            :, self.franka_ee_index - 1, :, :7
        ]  # -1 due to finxed base link.
        # Prepare mass matrix tensor
        # For franka, tensor shape is (num_envs, 7 + 2, 7 + 2), 2 for grippers.
        # _massmatrix = self.isaac_gym.acquire_mass_matrix_tensor(self.sim, "franka")
        # self.mm = gymtorch.wrap_tensor(_massmatrix)

    def april_coord_to_sim_coord(self, april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return self.april_to_sim_mat @ april_coord_mat

    def sim_coord_to_april_coord(self, sim_coord_mat):
        return self.sim_to_april_mat @ sim_coord_mat

    @property
    def april_to_sim_mat(self):
        return self.franka_from_origin_mat @ self.base_tag_from_robot_mat

    @property
    def sim_to_april_mat(self):
        return torch.tensor(
            np.linalg.inv(self.base_tag_from_robot_mat)
            @ np.linalg.inv(self.franka_from_origin_mat),
            device=self.device,
        )

    @property
    def sim_to_robot_mat(self):
        return torch.tensor(self.franka_from_origin_mat, device=self.device)

    @property
    def april_to_robot_mat(self):
        return torch.tensor(self.base_tag_from_robot_mat, device=self.device)

    @property
    def robot_to_ee_mat(self):
        return torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=self.device)

    @property
    def action_space(self):
        # Action space to be -1.0 to 1.0.
        if self.act_rot_repr == "quat":
            pose_dim = 7
        elif self.act_rot_repr == "rot_6d":
            pose_dim = 9
        else:  # axis
            pose_dim = 6

        low = np.array([-1] * pose_dim + [-1], dtype=np.float32)
        high = np.array([1] * pose_dim + [1], dtype=np.float32)

        low = np.tile(low, (self.num_envs, 1))
        high = np.tile(high, (self.num_envs, 1))

        return gym.spaces.Box(low, high, (self.num_envs, pose_dim + 1))

    @property
    def action_dimension(self):
        return self.action_space.shape[-1]

    @property
    def observation_space(self):
        low, high = -np.inf, np.inf
        parts_poses = self.furniture.num_parts * self.pose_dim
        img_size = reversed(self.img_size)
        img_shape = (3, *img_size) if self.channel_first else (*img_size, 3)

        obs_dict = {}
        robot_state = {}
        robot_state_dim = 0
        for k in self.obs_keys:
            if k.startswith("robot_state"):
                obs_key = k.split("/")[1]
                obs_shape = (ROBOT_STATE_DIMS[obs_key],)
                robot_state_dim += ROBOT_STATE_DIMS[obs_key]
                robot_state[obs_key] = gym.spaces.Box(low, high, obs_shape)
            elif k.startswith("color"):
                obs_dict[k] = gym.spaces.Box(0, 255, img_shape)
            elif k.startswith("depth"):
                obs_dict[k] = gym.spaces.Box(0, 255, img_size)
            elif k == "parts_poses":
                obs_dict[k] = gym.spaces.Box(low, high, (parts_poses,))
            else:
                raise ValueError(f"FurnitureSim does not support observation ({k}).")

        if robot_state:
            if self.concat_robot_state:
                obs_dict["robot_state"] = gym.spaces.Box(low, high, (robot_state_dim,))
            else:
                obs_dict["robot_state"] = gym.spaces.Dict(robot_state)

        return gym.spaces.Dict(obs_dict)

    @torch.no_grad()
    def step(self, action):
        """Robot takes an action.

        Args:
            action:
                (num_envs, 8): End-effector delta in [x, y, z, qx, qy, qz, qw, gripper] if self.act_rot_repr == "quat".
                (num_envs, 10): End-effector delta in [x, y, z, 6D rotation, gripper] if self.act_rot_repr == "rot_6d".
                (num_envs, 7): End-effector delta in [x, y, z, ax, ay, az, gripper] if self.act_rot_repr == "axis".
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(device=self.device)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        # Clip the action to be within the action space.
        action = torch.clamp(action, self.act_low, self.act_high)

        # Set the goal
        ee_pos, ee_quat_xyzw = self.get_ee_pose()

        # Move ee_quat real to the first element.
        ee_quat_wxyz = C.quat_xyzw_to_wxyz(ee_quat_xyzw)

        if self.act_rot_repr == "quat":
            # Real part is the last element in the quaternion.
            action_quat_xyzw = action[:, 3:7]

            # Move the real part to the first element.
            action_quat_wxyz = C.quat_xyzw_to_wxyz(action_quat_xyzw)

        elif self.act_rot_repr == "rot_6d":
            rot_6d = action[:, 3:9]
            rot_mat = pt.rotation_6d_to_matrix(rot_6d)
            # Real part is the first element in the quaternion.
            action_quat_wxyz = pt.matrix_to_quaternion(rot_mat)

        else:
            # Convert axis angle to quaternion.
            action_quat_wxyz = pt.matrix_to_quaternion(
                pt.axis_angle_to_matrix(action[:, 3:6])
            )

        if self.action_type == "delta":
            goals_pos = action[:, :3] + ee_pos
            goals_quat_wxyz = pt.quaternion_multiply(ee_quat_wxyz, action_quat_wxyz)
        elif self.action_type == "pos":
            goals_pos = action[:, :3]
            goals_quat_wxyz = action_quat_wxyz

        goals_quat_xyzw = C.quat_wxyz_to_xyzw(goals_quat_wxyz)

        # TODO: See if it makes sense to implement batching for the OSC controller.
        self.step_ctrl.set_goal(goals_pos, goals_quat_xyzw)

        pos_action = torch.zeros_like(self.dof_pos)
        torque_action = torch.zeros_like(self.dof_pos)
        grip_action = torch.zeros((self.num_envs, 1))

        grasp = action[:, -1]
        grip_sep = torch.where(
            (torch.sign(grasp) != torch.sign(self.last_grasp))
            & (torch.abs(grasp) > self.grasp_margin),
            torch.where(grasp < 0, self.max_gripper_width, torch.zeros_like(grasp)),
            torch.where(
                self.last_grasp < 0,
                self.max_gripper_width,
                torch.zeros_like(self.last_grasp),
            ),
        )
        self.last_grasp = grasp
        grip_action[:, -1] = grip_sep

        ee_pos, ee_quat = self.get_ee_pose()
        state_dict = {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "joint_positions": self.dof_pos[:, :7],
            "jacobian_diffik": self.jacobian_eef,
        }

        gripper_action_mask = (grip_sep > 0).unsqueeze(1)

        torque_action[:, 7:9] = torch.where(
            gripper_action_mask,
            sim_config["robot"]["gripper_torque"],
            -sim_config["robot"]["gripper_torque"],
        )

        pos_action[:, :7] = self.step_ctrl(state_dict)["joint_positions"]
        pos_action[:, 7:9] = torch.where(
            gripper_action_mask,
            self.max_gripper_width / 2,
            torch.zeros_like(pos_action[:, 7:9]),
        )
        self.isaac_gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(pos_action)
        )
        self.isaac_gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torque_action)
        )

        for _ in range(self.sim_steps):
            self.isaac_gym.simulate(self.sim)

        self.isaac_gym.fetch_results(self.sim, True)

        if not self.headless or self.render_cameras:
            self.isaac_gym.step_graphics(self.sim)

        # Refresh tensors.
        self.isaac_gym.refresh_dof_state_tensor(self.sim)
        self.isaac_gym.refresh_dof_force_tensor(self.sim)
        self.isaac_gym.refresh_rigid_body_state_tensor(self.sim)
        self.isaac_gym.refresh_jacobian_tensors(self.sim)

        # Update viewer
        if not self.headless:
            if self.ee_laser:
                # draw lines
                for _ in range(3):
                    noise = (np.random.random(3) - 0.5).astype(np.float32).reshape(
                        1, 3
                    ) * 0.001
                    offset = self.franka_from_origin_mat[:-1, -1].reshape(1, 3)
                    ee_z_axis = C.quat2mat(ee_quat[0]).cpu().numpy()[:, 2].reshape(1, 3)
                    line_start = ee_pos[0].cpu().numpy().reshape(1, 3) + offset + noise

                    # Move the start point higher
                    line_start = line_start - ee_z_axis * 0.019

                    line_end = line_start + ee_z_axis
                    lines = np.concatenate([line_start, line_end], axis=0)
                    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                    self.isaac_gym.add_lines(
                        self.viewer, self.envs[0], 1, lines, colors
                    )
            self.isaac_gym.draw_viewer(self.viewer, self.sim, False)
            self.isaac_gym.sync_frame_time(self.sim)
            self.isaac_gym.clear_lines(self.viewer)

        obs = self._get_observation()

        reward = self._reward()
        self.env_steps += 1

        return (
            obs,
            reward,
            self._done(),
            {"obs_success": True, "action_success": True},
        )

    def _reward(self):
        """Reward is 1 if two parts are assembled."""
        rewards = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )

        if self.manual_label:
            # Return zeros since the reward is manually labeled by data_collector.py.
            return rewards

        # Don't have to convert to AprilTag coordinate since the reward is computed with relative poses.
        parts_poses, founds = self._get_parts_poses(sim_coord=True)
        for env_idx in range(self.num_envs):
            env_parts_poses = parts_poses[env_idx].cpu().numpy()
            env_founds = founds[env_idx].cpu().numpy()
            rewards[env_idx] = self.furnitures[env_idx].compute_assemble(
                env_parts_poses, env_founds
            )

        if self.np_step_out:
            return rewards.cpu().numpy()

        return rewards

    def step_noop(self):
        """Take a no-op step."""

        # If we're doing delta control, we can simply apply a noop action:
        if self.action_type == "delta":
            noop = {
                "quat": torch.tensor(
                    [0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.float32, device=self.device
                ),
                "rot_6d": torch.tensor(
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                    dtype=torch.float32,
                    device=self.device,
                ),
            }[self.act_rot_repr]
            return self.step(noop)

        # Otherwise, we apply a noop action by setting temporily changing the control mode to delta control
        self.action_type = "delta"
        obs = self.step_noop()
        self.action_type = "pos"
        return obs

    def _get_parts_poses(self, sim_coord=False):
        """Get furniture parts poses in the AprilTag frame.

        Args:
            sim_coord: If True, return the poses in the simulator coordinate. Otherwise, return the poses in the AprilTag coordinate.

        Returns:
            parts_poses: (num_envs, num_parts * pose_dim). The poses of all parts in the AprilTag frame.
            founds: (num_envs, num_parts). Always 1 since we don't use AprilTag for detection in simulation.
        """
        founds = torch.ones(
            (self.num_envs, len(self.furniture.parts)),
            dtype=torch.float32,
            device=self.device,
        )
        # return parts_poses, founds
        if sim_coord:
            parts_poses = torch.zeros(
                (self.num_envs, len(self.furniture.parts) * self.pose_dim),
                dtype=torch.float32,
                device=self.device,
            )
            # Return the poses in the simulator coordinate.
            for part_idx in range(len(self.furniture.parts)):
                part = self.furniture.parts[part_idx]
                rb_idx = self.part_idxs[part.name]
                part_pose = self.rb_states[rb_idx, :7]
                parts_poses[
                    :, part_idx * self.pose_dim : (part_idx + 1) * self.pose_dim
                ] = part_pose[:, : self.pose_dim]

            return parts_poses, founds

        # Convert poses to AprilTag coordinate.
        # TODO: This can be optimized by not using Python list comprehension.
        rb_indices = torch.stack(
            [torch.tensor(self.part_idxs[part.name]) for part in self.furniture.parts],
            dim=0,
        ).T
        part_poses = self.rb_states[rb_indices, :7]

        part_poses_mat = C.pose2mat_batched(
            part_poses[:, :, :3], part_poses[:, :, 3:7], device=self.device
        )

        april_coord_poses_mat = self.sim_coord_to_april_coord(part_poses_mat)
        april_coord_poses = torch.cat(C.mat2pose_batched(april_coord_poses_mat), dim=-1)
        parts_poses = april_coord_poses.view(self.num_envs, -1)

        return parts_poses, founds

    def get_parts_poses(self, sim_coord=False):
        return self._get_parts_poses(sim_coord=sim_coord)

    def _save_camera_input(self):
        """Saves camera images to png files for debugging."""
        root = "sim_camera"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        Path(root).mkdir(exist_ok=True)

        for cam, handles in self.camera_handles.items():
            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_COLOR,
                f"{root}/{timestamp}_{cam}_sim.png",
            )

            self.isaac_gym.write_camera_image_to_file(
                self.sim,
                self.envs[0],
                handles[0],
                gymapi.IMAGE_DEPTH,
                f"{root}/{timestamp}_{cam}_sim_depth.png",
            )

    def _read_robot_state(self):
        joint_positions = self.dof_pos[:, :7]
        joint_velocities = self.dof_vel[:, :7]
        joint_torques = self.forces
        ee_pos, ee_quat = self.get_ee_pose()

        # Make sure the real part of the quaternion is positive.
        negative_mask = ee_quat[:, 3] < 0
        ee_quat[negative_mask] *= -1

        ee_pos_vel = self.rb_states[self.ee_idxs, 7:10]
        ee_ori_vel = self.rb_states[self.ee_idxs, 10:]
        gripper_width = self.gripper_width()

        robot_state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques": joint_torques,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_pos_vel": ee_pos_vel,
            "ee_ori_vel": ee_ori_vel,
            "gripper_width": gripper_width,
        }
        return {k: robot_state_dict[k] for k in self.robot_state_keys}

    def refresh(self):
        self.isaac_gym.simulate(self.sim)
        self.isaac_gym.fetch_results(self.sim, True)

        if not self.headless or self.render_cameras:
            self.isaac_gym.step_graphics(self.sim)

        # Refresh tensors.
        self.isaac_gym.refresh_dof_state_tensor(self.sim)
        self.isaac_gym.refresh_dof_force_tensor(self.sim)
        self.isaac_gym.refresh_rigid_body_state_tensor(self.sim)
        self.isaac_gym.refresh_jacobian_tensors(self.sim)

        if self.render_cameras:
            self.isaac_gym.render_all_camera_sensors(self.sim)
            self.isaac_gym.start_access_image_tensors(self.sim)

    def init_ctrl(self):
        self.step_ctrl = diffik_factory(
            real_robot=False,
            pos_scalar=self.pos_scalar,
            rot_scalar=self.rot_scalar,
        )
        self.ctrl_started = True

    def get_ee_pose(self):
        """Gets end-effector pose in world coordinate."""
        hand_pos = self.rb_states[self.ee_idxs, :3]
        hand_quat = self.rb_states[self.ee_idxs, 3:7]
        base_pos = self.rb_states[self.base_idxs, :3]
        base_quat = self.rb_states[self.base_idxs, 3:7]  # Align with world coordinate.
        return hand_pos - base_pos, hand_quat

    def gripper_width(self):
        return self.dof_pos[:, 7:8] + self.dof_pos[:, 8:9]

    def _done(self) -> torch.Tensor:
        dones = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        if self.manual_done:
            return dones
        for env_idx in range(self.num_envs):
            timeout = self.env_steps[env_idx] > self.furniture.max_env_steps
            if self.furnitures[env_idx].all_assembled() or timeout:
                dones[env_idx] = 1
                # if timeout:
                #     gym.logger.warn(f"[env] env_idx: {env_idx} timeout")
        if self.np_step_out:
            dones = dones.cpu().numpy().astype(bool)
        return dones

    def _get_color_obs(self, color_obs):
        color_obs = torch.stack(color_obs)[..., :-1]  # RGBA -> RGB
        if self.channel_first:
            color_obs = color_obs.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return color_obs

    def get_front_projection_view_matrix(self):
        cam_pos = self.front_cam_pos
        cam_target = self.front_cam_target
        width = self.img_size[0]
        height = self.img_size[1]
        near_plane = self.camera_cfg.near_plane
        far_plane = self.camera_cfg.far_plane
        horizontal_fov = self.camera_cfg.horizontal_fov

        # Compute aspect ratio
        aspect_ratio = width / height
        # Convert horizontal FOV from degrees to radians and calculate focal length
        fov_rad = np.radians(horizontal_fov)
        f = 1 / np.tan(fov_rad / 2)
        # Construct the projection matrix
        # fmt: off
        P = np.array(
            [
                [f / aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (far_plane + near_plane) / (near_plane - far_plane), (2 * far_plane * near_plane) / (near_plane - far_plane)],
                [0, 0, -1, 0],
            ]
        )
        # fmt: on

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        forward = normalize(cam_target - cam_pos)
        up = np.array([0, 1, 0])
        right = normalize(np.cross(up, forward))
        # Recompute Up Vector
        up = np.cross(forward, right)

        # Construct the View Matrix
        # fmt: off
        V = np.matrix(
            [
                [right[0], right[1], right[2], -np.dot(right, cam_pos)],
                [up[0], up[1], up[2], -np.dot(up, cam_pos)],
                [forward[0], forward[1], forward[2], -np.dot(forward, cam_pos)],
                [0, 0, 0, 1],
            ]
        )
        # fmt: on

        return P, V

    def _get_observation(self):
        robot_state = self._read_robot_state()
        if self.render_cameras:
            self.isaac_gym.render_all_camera_sensors(self.sim)
            self.isaac_gym.start_access_image_tensors(self.sim)
            color_obs = {
                k: self._get_color_obs(v)
                for k, v in self.camera_obs.items()
                if "color" in k
            }
            self.isaac_gym.end_access_image_tensors(self.sim)

        if robot_state and self.concat_robot_state:
            robot_state = torch.cat(list(robot_state.values()), -1)

        obs = {}
        if (
            isinstance(robot_state, (np.ndarray, torch.Tensor)) or robot_state
        ):  # Check if robot_state is empty.
            if self.robot_state_as_dict:
                obs["robot_state"] = robot_state
            else:
                obs.update(robot_state)  # Flatten the dict.
        for k in self.obs_keys:
            if k == "parts_poses":
                # Part poses in AprilTag.
                parts_poses, _ = self._get_parts_poses(sim_coord=False)
                obs["parts_poses"] = parts_poses

            elif k.startswith("color"):
                obs[k] = color_obs[k]

        return obs

    def get_observation(self):
        return self._get_observation()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError
        return self._get_observation()["color_image2"]

    def is_success(self):
        return [
            {"task": self.furnitures[env_idx].all_assembled()}
            for env_idx in range(self.num_envs)
        ]

    def reset(self):
        print("In orignal reset")
        for i in range(self.num_envs):
            self.reset_env(i)
            self.refresh()

        self.furniture.reset()

        self.refresh()
        self.assemble_idx = 0

        if self.save_camera_input:
            self._save_camera_input()

        return self._get_observation()

    def reset_env(self, env_idx, reset_franka=True, reset_parts=True):
        """Resets the environment. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            reset_franka: If True, then reset the franka for this env
            reset_parts: If True, then reset the part poses for this env
        """
        furniture: Furniture = self.furnitures[env_idx]
        furniture.reset()
        if self.randomness == Randomness.LOW and not self.init_assembled:
            furniture.randomize_init_pose(
                self.from_skill, pos_range=[-0.015, 0.015], rot_range=15
            )

        if self.randomness == Randomness.MEDIUM:
            furniture.randomize_init_pose(self.from_skill)
        elif self.randomness == Randomness.HIGH:
            furniture.randomize_high(self.high_random_idx)

        if reset_franka:
            self._reset_franka(env_idx)
        if reset_parts:
            self._reset_parts(env_idx)
        self.env_steps[env_idx] = 0
        self.move_neutral = False

    def reset_to(self, state):
        """Reset to a specific state.

        Args:
            state: List of observation dictionary for each environment.
        """
        for i in range(self.num_envs):
            self.reset_env_to(i, state[i])

    def reset_env_to(self, env_idx, state):
        """Reset to a specific state. **MUST refresh in between multiple calls
        to this function to have changes properly reflected in each environment.
        Also might want to set a zero-torque action via .set_dof_actuation_force_tensor
        to avoid additional movement**

        Args:
            env_idx: Environment index.
            state: A dict containing the state of the environment.
        """
        self.furnitures[env_idx].reset()
        dof_pos = np.concatenate(
            [
                state["robot_state"]["joint_positions"],
                np.array([state["robot_state"]["gripper_width"] / 2] * 2),
            ],
        )
        self._reset_franka(env_idx, dof_pos)
        self._reset_parts(env_idx, state["parts_poses"])
        self.env_steps[env_idx] = 0
        self.move_neutral = False

    def _update_franka_dof_state_buffer(self, dof_pos=None):
        """
        Sets internal tensor state buffer for Franka actor
        """
        # Low randomness only.
        if self.from_skill >= 1:
            dof_pos = torch.from_numpy(self.default_dof_pos)
            ee_pos = torch.from_numpy(
                self.furniture.furniture_conf["ee_pos"][self.from_skill]
            )
            ee_quat = torch.from_numpy(
                self.furniture.furniture_conf["ee_quat"][self.from_skill]
            )
            dof_pos = self.robot_model.inverse_kinematics(ee_pos, ee_quat)
        else:
            dof_pos = self.default_dof_pos if dof_pos is None else dof_pos

        # Views for self.dof_states (used with set_dof_state_tensor* function)
        self.dof_pos[:, 0 : self.franka_num_dofs] = torch.tensor(
            dof_pos, device=self.device, dtype=torch.float32
        )
        self.dof_vel[:, 0 : self.franka_num_dofs] = torch.tensor(
            [0] * len(self.default_dof_pos), device=self.device, dtype=torch.float32
        )

    def _reset_franka(self, env_idx, dof_pos=None):
        """
        Resets Franka actor within a single env. If calling multiple times,
        need to refresh in between calls to properly register individual env changes,
        and set zero torques on frankas across all envs to prevent the reset arms
        from moving while others are still being reset
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)

        # Update a single actor
        actor_idx = self.franka_actor_idxs_all_t[env_idx].reshape(1, 1)
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_idx),
            len(actor_idx),
        )

    def _reset_franka_all(self, dof_pos=None):
        """
        Resets all Franka actors across all envs
        """
        self._update_franka_dof_state_buffer(dof_pos=dof_pos)

        # Update all actors across envs at once
        self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(self.franka_actor_idxs_all_t),
            len(self.franka_actor_idxs_all_t),
        )

    def _reset_parts(self, env_idx, parts_poses=None, skip_set_state=False):
        """Resets furniture parts to the initial pose.

        Args:
            env_idx (int): The index of the environment.
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for part_idx, part in enumerate(self.furnitures[env_idx].parts):
            # Use the given pose.
            if parts_poses is not None:
                part_pose = parts_poses[part_idx * 7 : (part_idx + 1) * 7]

                pos = part_pose[:3]
                ori = T.to_homogeneous(
                    [0, 0, 0], T.quat2mat(part_pose[3:])
                )  # Dummy zero position.
            else:
                pos, ori = self._get_reset_pose(part)

            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = gymapi.Transform()
            part_pose.p = gymapi.Vec3(
                part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
            )
            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
            idxs = self.parts_handles[part.name]
            idxs = torch.tensor(idxs, device=self.device, dtype=torch.int32)

            self.root_pos[env_idx, idxs] = torch.tensor(
                [part_pose.p.x, part_pose.p.y, part_pose.p.z], device=self.device
            )
            self.root_quat[env_idx, idxs] = torch.tensor(
                [part_pose.r.x, part_pose.r.y, part_pose.r.z, part_pose.r.w],
                device=self.device,
            )

        if skip_set_state:
            # Set the value for the root state tensor, but don't call isaac gym function yet (useful when resetting all at once)
            # If skip_set_state == True, then must self.refresh() to register the isaac set_actor_root_state* function
            return

        # Reset root state for actors in a single env
        part_actor_idxs = torch.tensor(
            self.part_actor_idx_by_env[env_idx], device=self.device, dtype=torch.int32
        )
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(part_actor_idxs),
            len(part_actor_idxs),
        )

    def _reset_parts_all(self, parts_poses=None):
        """Resets ALL furniture parts to the initial pose.

        Args:
            parts_poses (np.ndarray): The poses of the parts. If None, the parts will be reset to the initial pose.
        """
        for env_idx in range(self.num_envs):
            self._reset_parts(env_idx, parts_poses=parts_poses, skip_set_state=True)

        # Reset root state for actors across all envs
        self.isaac_gym.get_sim_actor_count(self.sim)
        self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(self.part_actor_idxs_all_t),
            len(self.part_actor_idxs_all_t),
        )

    def _import_base_tag_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        base_asset_file = "furniture/urdf/base_tag.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, base_asset_file, asset_options
        )

    def _import_obstacle_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        obstacle_asset_file = "furniture/urdf/obstacle.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, obstacle_asset_file, asset_options
        )

    def _import_background_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        background_asset_file = "furniture/urdf/background.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, background_asset_file, asset_options
        )

    def _import_table_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset_file = "furniture/urdf/table.urdf"
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, table_asset_file, asset_options
        )

    def _import_franka_asset(self):
        self.franka_asset_file = (
            "franka_description_ros/franka_description/robots/franka_panda.urdf"
        )
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        return self.isaac_gym.load_asset(
            self.sim, ASSET_ROOT, self.franka_asset_file, asset_options
        )

    def get_assembly_action(self) -> torch.Tensor:
        """Scripted furniture assembly logic.

        Returns:
            Tuple (action for the assembly task, skill complete mask)
        """
        assert self.num_envs == 1  # Only support one environment for now.
        if self.furniture_name not in ["one_leg", "cabinet", "lamp", "round_table"]:
            raise NotImplementedError(
                "[one_leg, cabinet, lamp, round_table] are supported for scripted agent"
            )

        if self.assemble_idx > len(self.furniture.should_be_assembled):
            return torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=self.device)

        ee_pos, ee_quat = self.get_ee_pose()
        gripper_width = self.gripper_width()
        ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()

        if self.move_neutral:
            if ee_pos[2] <= 0.15 - 0.01:
                gripper = torch.tensor([-1], dtype=torch.float32, device=self.device)
                goal_pos = torch.tensor(
                    [ee_pos[0], ee_pos[1], 0.15], device=self.device
                )
                delta_pos = goal_pos - ee_pos
                delta_quat = torch.tensor([0, 0, 0, 1], device=self.device)
                action = torch.concat([delta_pos, delta_quat, gripper])
                return action.unsqueeze(0), 0
            else:
                self.move_neutral = False
        part_idx1, part_idx2 = self.furniture.should_be_assembled[self.assemble_idx]

        part1 = self.furniture.parts[part_idx1]
        part1_name = self.furniture.parts[part_idx1].name
        part1_pose = C.to_homogeneous(
            self.rb_states[self.part_idxs[part1_name]][0][:3],
            C.quat2mat(self.rb_states[self.part_idxs[part1_name]][0][3:7]),
        )
        part2 = self.furniture.parts[part_idx2]
        part2_name = self.furniture.parts[part_idx2].name
        part2_pose = C.to_homogeneous(
            self.rb_states[self.part_idxs[part2_name]][0][:3],
            C.quat2mat(self.rb_states[self.part_idxs[part2_name]][0][3:7]),
        )
        rel_pose = torch.linalg.inv(part1_pose) @ part2_pose
        assembled_rel_poses = self.furniture.assembled_rel_poses[(part_idx1, part_idx2)]
        if self.furniture.assembled(rel_pose.cpu().numpy(), assembled_rel_poses):
            self.assemble_idx += 1
            self.move_neutral = True
            return (
                torch.tensor(
                    [0, 0, 0, 0, 0, 0, 1, -1], dtype=torch.float32, device=self.device
                ).unsqueeze(0),
                1,
            )  # Skill complete is always 1 when assembled.
        if not part1.pre_assemble_done:
            goal_pos, goal_ori, gripper, skill_complete = part1.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
            )
        elif not part2.pre_assemble_done:
            goal_pos, goal_ori, gripper, skill_complete = part2.pre_assemble(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
            )
        else:
            goal_pos, goal_ori, gripper, skill_complete = self.furniture.parts[
                part_idx2
            ].fsm_step(
                ee_pos,
                ee_quat,
                gripper_width,
                self.rb_states,
                self.part_idxs,
                self.sim_to_april_mat,
                self.april_to_robot_mat,
                self.furniture.parts[part_idx1].name,
            )

        delta_pos = goal_pos - ee_pos

        # Scale translational action.
        delta_pos_sign = delta_pos.sign()
        delta_pos = torch.abs(delta_pos) * 2
        for i in range(3):
            if delta_pos[i] > 0.03:
                delta_pos[i] = 0.03 + (delta_pos[i] - 0.03) * np.random.normal(1.5, 0.1)
        delta_pos = delta_pos * delta_pos_sign

        # Clamp too large action.
        max_delta_pos = 0.11 + 0.01 * torch.rand(3, device=self.device)
        max_delta_pos[2] -= 0.04
        delta_pos = torch.clamp(delta_pos, min=-max_delta_pos, max=max_delta_pos)

        delta_quat = C.quat_mul(C.quat_conjugate(ee_quat), goal_ori)
        # Add random noise to the action.
        if (
            self.furniture.parts[part_idx2].state_no_noise()
            and np.random.random() < 0.50
        ):
            delta_pos = torch.normal(delta_pos, 0.005)
            delta_quat = C.quat_multiply(
                delta_quat,
                torch.tensor(
                    T.axisangle2quat(
                        [
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                            np.radians(np.random.normal(0, 5)),
                        ]
                    ),
                    device=self.device,
                ),
            ).to(self.device)
        action = torch.concat([delta_pos, delta_quat, gripper])
        return action.unsqueeze(0), skill_complete

    def assembly_success(self):
        return self._done().squeeze()

    def __del__(self):
        if not self.headless:
            self.isaac_gym.destroy_viewer(self.viewer)
        self.isaac_gym.destroy_sim(self.sim)

        if self.record:
            self.video_writer.release()


class FurnitureRLSimEnv(FurnitureSimEnv):
    """FurnitureSim environment for Reinforcement Learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ## Need these indices to reset position of the actors/parts
        # Store the default initialization pose for the parts in a convenient tensor
        self.parts_idx_list = torch.tensor(
            [self.parts_handles[part.name] for part in self.furniture.parts],
            device=self.device,
            dtype=torch.int32,
        )
        self.part_actor_idx_all = torch.tensor(
            [self.part_actor_idx_by_env[i] for i in range(self.num_envs)],
            device=self.device,
            dtype=torch.int32,
        )

        # Concat on the obstacle actor indices
        self.part_actor_idx_all = torch.cat(
            [self.part_actor_idx_all, self.obstacle_actor_idxs_by_env], dim=1
        )

        self.parts_initial_pos = torch.zeros(
            (len(self.parts_handles), 3), device=self.device
        )
        self.parts_initial_ori = torch.zeros(
            (len(self.parts_handles), 4), device=self.device
        )

        for i, part in enumerate(self.furniture.parts):
            pos, ori = self._get_reset_pose(part)
            part_pose_mat = self.april_coord_to_sim_coord(get_mat(pos, [0, 0, 0]))
            part_pose = gymapi.Transform()
            part_pose.p = gymapi.Vec3(
                part_pose_mat[0, 3], part_pose_mat[1, 3], part_pose_mat[2, 3]
            )
            reset_ori = self.april_coord_to_sim_coord(ori)
            part_pose.r = gymapi.Quat(*T.mat2quat(reset_ori[:3, :3]))
            idxs = self.parts_handles[part.name]
            idxs = torch.tensor(idxs, device=self.device, dtype=torch.int32)

            self.parts_initial_pos[i] = torch.tensor(
                [part_pose.p.x, part_pose.p.y, part_pose.p.z], device=self.device
            )
            self.parts_initial_ori[i] = torch.tensor(
                [part_pose.r.x, part_pose.r.y, part_pose.r.z, part_pose.r.w],
                device=self.device,
            )

        self.parts_initial_pos = self.parts_initial_pos.unsqueeze(0)
        self.parts_initial_ori = self.parts_initial_ori.unsqueeze(0)

        # Get the same for the obstacle actor
        self.obstacle_initial_pos = torch.tensor(
            [
                [
                    self.obstacle_reset_pose.p.x,
                    self.obstacle_reset_pose.p.y,
                    self.obstacle_reset_pose.p.z,
                ]
            ],
            device=self.device,
        ).unsqueeze(0)

        ## Need these indices to apply forces to the rigid bodies/parts
        self.rigid_body_count = self.isaac_gym.get_sim_rigid_body_count(self.sim)
        self.rigid_body_index_by_env = torch.zeros(
            (self.num_envs, len(self.furniture.parts)),
            dtype=torch.int32,
            device=self.device,
        )

        for i, part in enumerate(self.furniture.parts):
            for env_idx in range(self.num_envs):
                part_idxs = self.part_idxs[part.name]
                self.rigid_body_index_by_env[env_idx, i] = part_idxs[env_idx]

        self.force_multiplier = torch.tensor(
            [25, 1, 1, 1, 1], device=self.device
        ).unsqueeze(-1)
        self.torque_multiplier = torch.tensor(
            [70, 1, 1, 1, 1], device=self.device
        ).unsqueeze(-1)

        self.max_force_magnitude = kwargs.get("max_force_magnitude", 0.2)
        self.max_torque_magnitude = kwargs.get("max_torque_magnitude", 0.005)

        # Randomize plus/minus 2 cm
        obstacle_range = 2

        self.obstacle_range_a = torch.tensor(
            [2 * obstacle_range, 2 * obstacle_range, 0], device=self.device
        )
        self.obstacle_range_b = torch.tensor(
            [obstacle_range, obstacle_range, 0], device=self.device
        )

        print(
            f"Max force magnitude: {self.max_force_magnitude}, Max torque magnitude: {self.max_torque_magnitude}"
        )

    def reset(self, env_idxs: torch.Tensor = None):
        # return super().reset()
        # can also reset the full set of robots/parts, without applying torques and refreshing
        if env_idxs is None:
            env_idxs = torch.arange(
                self.num_envs, device=self.device, dtype=torch.int32
            )

        assert env_idxs.numel() > 0, "env_idxs must have at least one element"

        for idx in env_idxs:
            self.furnitures[idx].reset()

        self._reset_frankas(env_idxs)
        self._reset_parts_multiple(env_idxs)
        self.env_steps[env_idxs] = 0

        self.refresh()

        return self._get_observation()

    def increment_randomness(self):
        force_magnitude_limit = 1
        torque_magnitude_limit = 0.05

        self.max_force_magnitude = min(
            self.max_force_magnitude + 0.01, force_magnitude_limit
        )
        self.max_torque_magnitude = min(
            self.max_torque_magnitude + 0.0005, torque_magnitude_limit
        )
        print(
            f"Increased randomness: F->{self.max_force_magnitude:.4f}, "
            f"T->{self.max_torque_magnitude:.4f}"
        )

    def _reset_frankas(self, env_idxs: torch.Tensor):
        dof_pos = self.default_dof_pos

        # Views for self.dof_states (used with set_dof_state_tensor* function)
        self.dof_pos[:, 0 : self.franka_num_dofs] = torch.tensor(
            dof_pos, device=self.device, dtype=torch.float32
        )
        self.dof_vel[:, 0 : self.franka_num_dofs] = torch.tensor(
            [0] * len(self.default_dof_pos), device=self.device, dtype=torch.float32
        )

        # Update a list of actors
        actor_idx = self.franka_actor_idxs_all_t[env_idxs].reshape(-1, 1)
        success = self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(actor_idx),
            len(actor_idx),
        )
        assert success, "Failed to set franka state"

    def _reset_parts_multiple(self, env_idxs):
        """Resets furniture parts to the initial pose."""
        ## Reset
        # Reset the parts to the initial pose
        self.root_pos[env_idxs.unsqueeze(1), self.parts_idx_list] = (
            self.parts_initial_pos.clone()
        )
        self.root_quat[env_idxs.unsqueeze(1), self.parts_idx_list] = (
            self.parts_initial_ori.clone()
        )

        # Find the position we want to place the obstacle at here
        # We randomize the obstacle here because we want it fixed and don't apply forces to it later
        # Sample x and y values in [-2, 2] that we want to add to the initial position
        obstacle_pos_offsets = (
            torch.rand((env_idxs.numel(), 1, 3), device=self.device)
            * self.obstacle_range_a
            + self.obstacle_range_b
        )
        self.root_pos[env_idxs.unsqueeze(1), self.obstacle_handle] = (
            self.obstacle_initial_pos.clone() + obstacle_pos_offsets
        )

        # Get the actor and rigid body indices for the parts in question
        part_actor_idxs = self.part_actor_idx_all[env_idxs].view(-1)

        # Update the sim state tensors
        success = self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(part_actor_idxs),
            len(part_actor_idxs),
        )

        assert success, "Failed to set part state"

        ## Random forces
        # Generate random forces in the xy plane for all parts across all environments
        part_rigid_body_idxs = self.rigid_body_index_by_env[env_idxs]
        force_theta = (
            torch.rand(part_rigid_body_idxs.shape + (1,), device=self.device)
            * 2
            * np.pi
        )
        force_magnitude = (
            torch.rand(part_rigid_body_idxs.shape + (1,), device=self.device)
            * self.max_force_magnitude
        )
        forces = torch.cat(
            [
                force_magnitude * torch.cos(force_theta),
                force_magnitude * torch.sin(force_theta),
                torch.zeros_like(force_magnitude),
            ],
            dim=-1,
        )
        # Scale the forces by the mass of the parts
        forces = (forces * self.force_multiplier).view(-1, 3)

        ## Random torques

        # Generate random torques for all parts across all environments in the z direction
        z_torques = self.max_torque_magnitude * (
            torch.rand(part_rigid_body_idxs.shape + (1,), device=self.device) * 2 - 1
        )
        torques = torch.cat(
            [
                torch.zeros_like(z_torques),
                torch.zeros_like(z_torques),
                z_torques,
            ],
            dim=-1,
        )

        # Create a tensor to hold forces for all rigid bodies
        all_forces = torch.zeros((self.rigid_body_count, 3), device=self.device)
        all_torques = torch.zeros((self.rigid_body_count, 3), device=self.device)
        part_rigid_body_idxs = part_rigid_body_idxs.view(-1)
        all_torques[part_rigid_body_idxs] = torques.view(-1, 3)
        all_forces[part_rigid_body_idxs] = forces.view(-1, 3)

        # Fill the appropriate indices with the generated forces
        # Apply the forces to the rigid bodies
        success = self.isaac_gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(all_forces),
            gymtorch.unwrap_tensor(all_torques),
            gymapi.GLOBAL_SPACE,  # Apply forces in the world space
        )

        assert success, "Failed to apply forces to parts"


class FurnitureRLSimEnvFinetune(FurnitureRLSimEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load in the normalizer weights
        self.normalizer = LinearNormalizer()
        self.normalizer.load_state_dict(
            torch.load(
                "/data/scratch/ankile/diffusion-adapt/data/furniture/one_leg_low_dim_normalizer.pth"
            )
        )
        self.normalizer.to(self.device)

    def reset_arg(self, options_list=None):
        obs = self.reset()

        # Concat together the robot_state and parts_poses
        robot_state = obs["robot_state"]

        # Convert the robot state to have 6D pose
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        parts_poses = obs["parts_poses"]

        obs = torch.cat([robot_state, parts_poses], dim=-1)
        nobs = self.normalizer(obs, "observations", forward=True)

        return nobs.cpu().numpy()

    def step(self, action):
        action = torch.tensor(action, device=self.device)

        # Denormalize the action
        action = self.normalizer(action, "actions", forward=False)

        obs, reward, done, info = super().step(action)
        robot_state = obs["robot_state"]

        # Convert the robot state to have 6D pose
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        parts_poses = obs["parts_poses"]

        obs = torch.cat([robot_state, parts_poses], dim=-1)
        nobs = self.normalizer(obs, "observations", forward=True)

        return (
            nobs.cpu().numpy(),
            reward.squeeze().cpu().numpy(),
            done.squeeze().cpu().numpy(),
            info,
        )


class FurnitureRLSimEnvPlaceTabletop(FurnitureRLSimEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define the goal position for the tabletop
        self.tabletop_goal = torch.tensor([0.0819, 0.2866, -0.0157], device=self.device)

    def _reward(self):
        """Calculates the reward for the current state of the environment."""
        # Get the end effector position
        parts_poses, _ = self._get_parts_poses(sim_coord=False)
        tabletop_pos = parts_poses[:, :3]

        reward = torch.zeros(self.num_envs, device=self.device)

        # Set the reward to be 1 if the distance is less than 0.1 (10 cm) from the goal
        reward[torch.norm(tabletop_pos - self.tabletop_goal, dim=-1) < 0.005] = 1

        return reward


class FurnitureRLSimEnvReacher(FurnitureRLSimEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define the goal position for the end effector
        self.goal_pos = torch.tensor([0.5934, -0.2813, 0.5098], device=self.device)

    def _reward(self):
        """Calculates the reward for the current state of the environment."""
        # Get the end effector position
        ee_pos, _ = self.get_ee_pose()

        reward = torch.zeros(self.num_envs, device=self.device)

        # Set the reward to be 1 if the distance is less than 0.1 (10 cm) from the goal
        reward[torch.norm(ee_pos - self.goal_pos, dim=-1) < 0.1] = 1

        return reward
