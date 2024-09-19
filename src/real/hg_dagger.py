# File: src/real/hg_dagger.py

import argparse
import time
from pathlib import Path
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import necessary modules from your codebase
from src.eval.eval_utils import get_model_from_api_or_cached
from src.dataset.rollout_buffer import RolloutBuffer
from src.models.residual import ResidualPolicy  # Adjust if necessary
from src.behavior.diffusion import DiffusionPolicy  # Adjust if necessary
from src.dataset.normalizer import LinearNormalizer
from src.common.pytorch_util import dict_to_device

# Rest of the imports from your real_teleop.py
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

from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

from src.real.serials import FRONT_CAMERA_SERIAL, WRIST_CAMERA_SERIAL

from ipdb import set_trace as bp

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

# Add new arguments
parser.add_argument("--trajectories_per_round", type=int, default=10)
parser.add_argument("--epochs_per_round", type=int, default=1)
parser.add_argument("--actor_wandb_id", type=str, required=True)
parser.add_argument("--actor_wt_type", type=str, default="latest")
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument(
    "--wandb_mode", type=str, default="disabled"
)  # 'online' or 'disabled'
parser.add_argument("--batch_size", type=int, default=32)

args = parser.parse_args()

poly_util = PolymetisHelper()

# The rest of your code remains the same until the main function


# Modify KeyboardInterface to include toggling human intervention
class CustomKeyboardInterface(KeyboardInterface):
    def __init__(self):
        super().__init__()
        self.intervention_on = True  # Start with human intervention on
        self.toggle_intervention_key = "i"

    def get_action(self):
        action, collect_enum = super().get_action()
        if self.kb.kbhit():
            c = self.kb.getch()
            c_ord = ord(c)
            if c == self.toggle_intervention_key:
                self.intervention_on = not self.intervention_on
                print(f"Human intervention {'on' if self.intervention_on else 'off'}")
        return action, collect_enum


# Modify ObsActHelper to include actor_model
class ObsActHelper:
    def __init__(
        self,
        sm,
        keyboard,
        robot,
        gripper,
        front_image_pipeline,
        wrist_image_pipeline,
        actor_model,
    ):

        self.sm = sm
        self.keyboard = keyboard

        self.robot = robot
        self.gripper = gripper

        self.front_image_pipeline = front_image_pipeline
        self.wrist_image_pipeline = wrist_image_pipeline

        self.actor_model = actor_model
        self.device = torch.device("cpu")  # Adjust if using GPU

        self._setup()

    # ... [Rest of the ObsActHelper code remains the same]

    def get_action(self):
        # get keyboard actions/flags
        keyboard_action, collect_enum = self.keyboard.get_action()

        # Check if human intervention is on
        if self.keyboard.intervention_on:
            # Use human action
            sm_state = self.sm.get_motion_state_transformed()

            # scale pos command
            dpos = (
                sm_state[:3]
                * (self.max_pos_speed / self.frequency)
                * self.sm_dpos_scalar
            )

            # convert and scale rot command
            drot_xyz = sm_state[3:] * (self.max_rot_speed / self.frequency)
            drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
            drot_rotvec *= self.sm_drot_scalar
            drot = st.Rotation.from_rotvec(drot_rotvec)

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

            # Make a delta action of xyz + quat_xyzw that we can scale before sending to robot
            delta_action = np.concatenate(
                [dpos, drot.as_quat(), np.array([self.grasp_flag])]
            )

            # overwrite action from keyboard action (for screwing)
            kb_taken = False
            if keyboard_action is not None and not (
                np.allclose(keyboard_action[3:6], 0.0)
            ):
                delta_action[3:7] = keyboard_action[3:7]
                kb_taken = True
                action_taken = True

            pos_bounds_m = 0.025
            ori_bounds_deg = 20

            delta_action = (
                scale_scripted_action(
                    torch.from_numpy(delta_action).unsqueeze(0),
                    pos_bounds_m=pos_bounds_m,
                    ori_bounds_deg=ori_bounds_deg,
                )
                .squeeze()
                .numpy()
            )

            # write out action
            new_target_pose = self.target_pose.copy()
            dpos, drot = delta_action[:3], st.Rotation.from_quat(delta_action[3:7])
            new_target_pose[:3] += dpos
            if kb_taken:
                # right multiply (more intuitive for screwing)
                new_target_pose[3:] = (
                    st.Rotation.from_rotvec(self.target_pose[3:]) * drot
                ).as_rotvec()
            else:
                # left multiply (more intuitive for spacemouse)
                new_target_pose[3:] = (
                    drot * st.Rotation.from_rotvec(self.target_pose[3:])
                ).as_rotvec()
            new_target_pose_mat = self.to_pose_mat(new_target_pose)
            current_pose_mat = self.to_pose_mat(self.target_pose)

            action_struct = ActionContainer(
                current_pose_mat=current_pose_mat,
                next_pose_mat=new_target_pose_mat,
                grasp_flag=self.grasp_flag,
                action_taken=action_taken,
                collect_enum=collect_enum,
                is_gripper_open=self.gripper_open,
                toggle_gripper=toggle_gripper,
            )
            return action_struct

        else:
            # Use policy action
            # Get observation
            observation = self.get_observation()

            # Preprocess observation as needed
            model_input = self.preprocess_observation(observation)

            # Convert to torch tensor and move to device
            model_input = {
                k: torch.tensor(v).unsqueeze(0).to(self.device)
                for k, v in model_input.items()
            }

            # Query the actor model
            with torch.no_grad():
                action_output = self.actor_model.action(model_input)

            # Convert action to delta_action
            delta_action = self.model_action_to_delta_action(action_output)

            action_taken = True

            # manage grasping
            self.grasp_flag = int(np.sign(delta_action[-1]))
            is_gripper_open = self.grasp_flag == 1
            toggle_gripper = self.gripper_open != is_gripper_open
            self.gripper_open = is_gripper_open

            # Compute new target pose
            new_target_pose = self.target_pose.copy()
            dpos, drot = delta_action[:3], st.Rotation.from_quat(delta_action[3:7])
            new_target_pose[:3] += dpos
            new_target_pose[3:] = (
                drot * st.Rotation.from_rotvec(self.target_pose[3:])
            ).as_rotvec()
            new_target_pose_mat = self.to_pose_mat(new_target_pose)
            current_pose_mat = self.to_pose_mat(self.target_pose)

            action_struct = ActionContainer(
                current_pose_mat=current_pose_mat,
                next_pose_mat=new_target_pose_mat,
                grasp_flag=self.grasp_flag,
                action_taken=action_taken,
                collect_enum=collect_enum,
                is_gripper_open=self.gripper_open,
                toggle_gripper=toggle_gripper,
            )
            return action_struct

    def preprocess_observation(self, observation):
        # Convert observation to model input
        # Example: extract robot_state and images if needed
        model_input = {}
        model_input["robot_state"] = observation["robot_state"]["ee_pos"]
        # Add image processing if needed
        return model_input

    def model_action_to_delta_action(self, action_output):
        # Convert model output to delta_action
        # Assuming action_output is a tensor of shape [1, action_dim]
        action = action_output.squeeze(0).cpu().numpy()
        delta_action = action  # Adjust if necessary
        return delta_action


# Main function
def main():
    # ... [Existing initialization code]

    # Load the actor model
    actor_cfg, actor_wts_path = get_model_from_api_or_cached(
        args.actor_wandb_id,
        wt_type=args.actor_wt_type,
        wandb_mode=args.wandb_mode,
    )

    # Initialize the actor model
    device = torch.device("cpu")  # Adjust if using GPU
    actor_model = DiffusionPolicy(device, actor_cfg)  # Adjust model class if necessary
    actor_wts = torch.load(actor_wts_path, map_location=device)
    if "model_state_dict" in actor_wts:
        actor_wts = actor_wts["model_state_dict"]
    actor_model.load_state_dict(actor_wts)
    actor_model.to(device)
    actor_model.eval()

    # Initialize optimizer for training the actor model
    optimizer = optim.AdamW(
        actor_model.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=1e-6,
    )

    # Initialize variables for data collection and training
    # We'll use RolloutBuffer from src.dataset.rollout_buffer
    # Assume action_dim and obs_dim are known
    action_dim = actor_model.action_dim  # Adjust as needed
    # For obs_dim, you may need to define based on observation keys
    obs_dim = ...  # Define based on your observation

    buffer = RolloutBuffer(
        max_size=100000,  # Adjust buffer size as needed
        state_dim=obs_dim,
        action_dim=action_dim,
        pred_horizon=1,
        obs_horizon=1,
        action_horizon=1,
        device=device,
        predict_past_actions=False,
        include_future_obs=False,
        include_images=False,
    )

    num_collected_trajectories = 0

    # Modify KeyboardInterface initialization
    keyboard = CustomKeyboardInterface()

    # Initialize ObsActHelper with actor_model
    # Wait to initialize it within the SharedMemoryManager context

    # ... [Rest of the setup code remains the same]

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
                actor_model=actor_model,
            )

            # obs_act_helper.set_target_pose(target_pose)
            obs_act_helper.set_target_pose(tip_target_pose)
            obs_act_helper.set_constants(
                max_pos_speed=args.max_pos_speed,
                max_rot_speed=args.max_rot_speed,
                sm_dpos_scalar=sm_dpos_scalar,
                sm_drot_scalar=sm_drot_scalar,
                frequency=args.frequency,
            )

            global_start_time = time.time()
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
                action_current_pose_mat = action_struct.current_pose_mat
                action_next_pose_mat = action_struct.next_pose_mat
                grasp_flag = action_struct.grasp_flag
                action_taken = action_struct.action_taken
                collect_enum = action_struct.collect_enum
                is_gripper_open = action_struct.is_gripper_open
                toggle_gripper = action_struct.toggle_gripper

                if collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                    num_collected_trajectories += 1
                    if num_collected_trajectories >= args.trajectories_per_round:
                        # Perform training
                        actor_model.train()
                        trainloader = DataLoader(
                            buffer,
                            batch_size=args.batch_size,
                            num_workers=0,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False,
                            persistent_workers=False,
                        )
                        for epoch in range(args.epochs_per_round):
                            for batch in trainloader:
                                optimizer.zero_grad()
                                batch = dict_to_device(batch, device)
                                loss = actor_model.compute_loss(batch)[0]
                                loss.backward()
                                optimizer.step()
                        actor_model.eval()
                        # Reset counters
                        num_collected_trajectories = 0
                        buffer.clear()
                    # Reset episode data
                    # ... [Reset any necessary variables]
                    continue  # Start new trajectory

                # send command to the robot
                robot.update_desired_ee_pose(
                    convert_tip2wrist(action_next_pose_mat), dt=dt
                )
                execute_gripper_action(toggle_gripper, is_gripper_open)

                # Collect data
                if action_taken:
                    # Convert action to the format expected by the buffer
                    action = obs_act_helper.to_isaac_dpose_from_abs(
                        current_pose_mat=action_current_pose_mat,
                        goal_pose_mat=action_next_pose_mat,
                        grasp_flag=grasp_flag,
                        rm=True,
                    )
                    # Preprocess observation and action
                    model_input = obs_act_helper.preprocess_observation(observation)
                    obs_tensor = np.concatenate(
                        [model_input[key] for key in model_input]
                    )
                    action_tensor = action

                    # Add to buffer
                    buffer.add_transition(
                        state=obs_tensor,
                        action=action_tensor,
                        reward=0.0,  # Reward is not used here
                        done=False,
                    )

                target_pose = polypose2target(robot.get_ee_pose())
                tip_target_pose = wrist_target_to_tip(target_pose)
                # obs_act_helper.set_target_pose(target_pose)
                obs_act_helper.set_target_pose(tip_target_pose)

                # Draw the current and target pose (in meshcat)
                mc_util.meshcat_frame_show(
                    mc_vis,
                    f"scene/target_pose_wrist",
                    convert_tip2wrist(action_next_pose_mat),
                )
                mc_util.meshcat_frame_show(
                    mc_vis, f"scene/target_pose_tip", action_next_pose_mat
                )
                mc_util.meshcat_frame_show(
                    mc_vis,
                    f"scene/current_pose",
                    poly_util.polypose2mat(robot.get_ee_pose()),
                )

                precise_wait(t_cycle_end)
                iter_idx += 1

    global_total_time = time.time() - global_start_time
    print(f"Time elapsed: {global_total_time}")


if __name__ == "__main__":
    main()
