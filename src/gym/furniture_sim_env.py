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


import time
from typing import Union
from datetime import datetime
from pathlib import Path

from furniture_bench.furniture.furniture import Furniture
import torch
import cv2
import gym
import numpy as np

import pytorch3d.transforms as pt

import furniture_bench.utils.transform as T
import furniture_bench.controllers.control_utils as C
from furniture_bench.utils.pose import get_mat

from ipdb import set_trace as bp
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv


class FurnitureRLSimEnv(FurnitureSimEnv):
    """FurnitureSim environment for Reinforcement Learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.initial_pos = torch.zeros((len(self.parts_handles), 3), device=self.device)
        self.initial_ori = torch.zeros((len(self.parts_handles), 4), device=self.device)

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

            self.initial_pos[i] = torch.tensor(
                [part_pose.p.x, part_pose.p.y, part_pose.p.z], device=self.device
            )
            self.initial_ori[i] = torch.tensor(
                [part_pose.r.x, part_pose.r.y, part_pose.r.z, part_pose.r.w],
                device=self.device,
            )

        # Lift the initial z position of the parts by 1 cm
        # self.initial_pos[:, 2] += 0.01

        self.initial_pos = self.initial_pos.unsqueeze(0)
        self.initial_ori = self.initial_ori.unsqueeze(0)

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

        self.max_force_magnitude = 0.2
        self.max_torque_magnitude = 0.005

    def reset(self, env_idxs: torch.tensor = None):
        # return super().reset()
        # can also reset the full set of robots/parts, without applying torques and refreshing
        if env_idxs is None:
            env_idxs = torch.arange(
                self.num_envs, device=self.device, dtype=torch.int32
            )

        assert env_idxs.numel() > 0, "env_idxs must have at least one element"

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

        # Reset the parts to the initial pose
        self.root_pos[env_idxs.unsqueeze(1), self.parts_idx_list] = (
            self.initial_pos.clone()
        )
        self.root_quat[env_idxs.unsqueeze(1), self.parts_idx_list] = (
            self.initial_ori.clone()
        )

        # Get the actor and rigid body indices for the parts in question
        part_rigid_body_idxs = self.rigid_body_index_by_env[env_idxs]
        part_actor_idxs = self.part_actor_idx_all[env_idxs].view(-1)

        ## Random forces
        # Generate random forces in the xy plane for all parts across all environments
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

        # Update the sim state tensors
        success = self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            gymtorch.unwrap_tensor(part_actor_idxs),
            len(part_actor_idxs),
        )

        assert success, "Failed to set part state"


class FurnitureRLSimEnvPlaceTabletop(FurnitureRLSimEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define the goal position for the tabletop
        self.tabletop_goal = torch.tensor([0.0819, 0.2866, -0.0157], device=self.device)

    def _reward(self):
        """Calculates the reward for the current state of the environment."""
        # Get the end effector position
        parts_poses, founds = self._get_parts_poses(sim_coord=False)
        tabletop_pos = parts_poses[:, :3]

        reward = torch.zeros(self.num_envs, device=self.device)

        # Set the reward to be 1 if the distance is less than 0.1 (10 cm) from the goal
        reward[torch.norm(tabletop_pos - self.tabletop_goal, dim=-1) < 0.005] = 1

        return reward
