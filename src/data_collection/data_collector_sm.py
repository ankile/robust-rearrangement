"""Define data collection class that rollout the environment, get action from the interface (e.g., teleoperation, automatic scripts), and save data."""
import time
import pickle
from datetime import datetime
from pathlib import Path

import gym
import torch
from tqdm import tqdm
from ipdb import set_trace as bp

from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.sim_config import sim_config
from src.data_processing.utils import resize, resize_crop
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action
from src.visualization.render_mp4 import pickle_data, unpickle_data


import os
import sys
import time
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy.spatial.transform as st
from furniture_bench.device.spacemouse.spacemouse_shared_memory import Spacemouse


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


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


class DataCollectorSpaceMouse:
    """Demonstration collection class.
    `pkl` files have resized images while `mp4` / `png` files save raw camera inputs.
    """

    def __init__(
        self,
        is_sim: bool,
        data_path: str,
        device_interface: DeviceInterface,
        furniture: str,
        headless: bool,
        draw_marker: bool,
        manual_label: bool,
        scripted: bool,
        randomness: Randomness.LOW,
        compute_device_id: int,
        graphics_device_id: int,
        pkl_only: bool = False,
        save_failure: bool = False,
        num_demos: int = 100,
        resize_sim_img: bool = True,
        verbose: bool = True,
        show_pbar: bool = False,
        ctrl_mode: str = "osc",
        ee_laser: bool = True,
        right_multiply_rot: bool = True,
        compress_pickles: bool = False,
    ):
        """
        Args:
            is_sim (bool): Whether to use simulator or real world environment.
            data_path (str): Path to save data.
            device_interface (DeviceInterface): Keyboard and/or Oculus interface.
            furniture (str): Name of the furniture.
            headless (bool): Whether to use headless mode.
            draw_marker (bool): Whether to draw AprilTag marker.
            manual_label (bool): Whether to manually label the reward.
            scripted (bool): Whether to use scripted function for getting action.
            randomness (str): Initialization randomness level.
            gpu_id (int): GPU ID.
            pkl_only (bool): Whether to save only `pkl` files (i.e., exclude *.mp4 and *.png).
            save_failure (bool): Whether to save failure trajectories.
            num_demos (int): The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
            ctrl_mode (str): 'osc' (joint torque, with operation space control) or 'diffik' (joint impedance, with differential inverse kinematics control)
            ee_laser (bool): If True, show a line coming from the end-effector in the viewer
            right_multiply_rot (bool): If True, convert rotation actions (delta rot) assuming they're applied as RIGHT multiplys (local rotations)
        """
        if is_sim:
            self.env = gym.make(
                "FurnitureSimFull-v0",
                furniture=furniture,
                max_env_steps=sim_config["scripted_timeout"][furniture]
                if scripted
                else 3000,
                headless=headless,
                num_envs=1,  # Only support 1 for now.
                manual_done=False if scripted else True,
                resize_img=resize_sim_img,
                np_step_out=False,  # Always output Tensor in this setting. Will change to numpy in this code.
                channel_first=False,
                randomness=randomness,
                compute_device_id=compute_device_id,
                graphics_device_id=graphics_device_id,
                ctrl_mode=ctrl_mode,
                ee_laser=ee_laser,
            )
        else:
            if randomness == "med":
                randomness = Randomness.MEDIUM_COLLECT
            elif randomness == "high":
                randomness = Randomness.HIGH_COLLECT

            self.env = gym.make(
                "FurnitureBench-v0",
                furniture=furniture,
                resize_img=False,
                manual_done=True,
                with_display=not headless,
                draw_marker=draw_marker,
                randomness=randomness,
            )

        self.is_sim = is_sim
        self.data_path = Path(data_path)
        self.device_interface = device_interface
        self.headless = headless
        self.manual_label = manual_label
        self.furniture = furniture
        self.num_demos = num_demos
        self.scripted = scripted

        self.traj_counter = 0
        self.num_success = 0
        self.num_fail = 0

        self.pkl_only = pkl_only
        self.save_failure = save_failure
        self.resize_sim_img = resize_sim_img
        self.compress_pickles = compress_pickles

        self.verbose = verbose
        self.pbar = None if not show_pbar else tqdm(total=self.num_demos)

        # our flags
        self.right_multiply_rot = right_multiply_rot

        self._reset_collector_buffer()

    def load_state(self, state: dict):
        # Get the state dict at the end of a one_leg trajectory
        print("Loading state")
        state = unpickle_data(
            "/data/scratch-oc40/pulkitag/ankile/furniture-data/raw/sim/one_leg/scripted/low/success/2024-01-22T01:13:22.pkl.xz"
        )["observations"][-1]
        self.env.reset_env_to(env_idx=0, state=state)
        self.env.refresh()

        return state

    def _squeeze_and_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        v = v.squeeze()
        return v

    def collect(self):
        self.verbose_print("[data collection] Start collecting the data!")

        from collections import namedtuple

        args = namedtuple(
            "Args",
            [
                "frequency",
                "command_latency",
                "deadzone",
                "max_pos_speed",
                "max_rot_speed",
            ],
        )

        args.frequency = 10
        args.command_latency = 0.01
        args.deadzone = 0.05
        if self.env.ctrl_mode == "diffik":
            args.max_pos_speed = 0.3
            args.max_rot_speed = 0.7
        else:
            args.max_pos_speed = 0.8
            # args.max_rot_speed = 2.5
            args.max_rot_speed = 4.0

        frequency = args.frequency
        dt = 1 / frequency
        command_latency = args.command_latency

        obs = self.reset()
        done = False

        translation, quat_xyzw = self.env.get_ee_pose()

        # Overwrite the state with our desired state
        # translation = torch.from_numpy(state["robot_state"]["ee_pos"]).float()
        # quat_xyzw = torch.from_numpy(state["robot_state"]["ee_quat"]).float()

        env_device = self.env.device
        translation, quat_xyzw = (
            translation.cpu().numpy().squeeze(),
            quat_xyzw.cpu().numpy().squeeze(),
        )
        gripper_width = self.env.gripper_width()
        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        target_pose_rv = np.array([*translation, *rotvec])
        gripper_open = gripper_width >= 0.06
        grasp_flag = torch.from_numpy(np.array([-1 if gripper_open else 1])).to(
            env_device
        )

        def pose_rv2mat(pose_rv):
            pose_mat = np.eye(4)
            pose_mat[:-1, -1] = pose_rv[:3]
            pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_rv[3:]).as_matrix()
            return pose_mat

        def to_isaac_dpose_from_abs(
            current_pose_mat, goal_pose_mat, grasp_flag, device, rm=True
        ):
            """
            Convert from absolute current and desired pose to delta pose

            Args:
                rm (bool): 'rm' stands for 'right multiplication' - If True, assume commands send as right multiply (local rotations)
            """
            if rm:
                delta_rot_mat = (
                    np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
                )
            else:
                delta_rot_mat = goal_pose_mat[:-1:-1] @ np.linalg.inv(
                    current_pose_mat[:-1, :-1]
                )

            dpos = goal_pose_mat[:-1, -1] - current_pose_mat[:-1, -1]
            target_translation = torch.from_numpy(dpos).float().to(device)

            target_rot = st.Rotation.from_matrix(delta_rot_mat)
            target_quat_xyzw = torch.from_numpy(target_rot.as_quat()).float().to(device)
            target_dpose = torch.cat(
                (target_translation, target_quat_xyzw, grasp_flag), dim=-1
            ).reshape(1, -1)
            return target_dpose

        target_pose_last_action_rv = None
        ready_to_grasp = True
        steps_since_grasp = 0

        with SharedMemoryManager() as shm_manager:
            with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
                t_start = time.monotonic()
                iter_idx = 0
                prev_keyboard_gripper = -1

                while self.num_success < self.num_demos:
                    if self.scripted:
                        raise ValueError("Not using scripted with spacemouse")

                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    # t_command_target = t_cycle_end + dt
                    precise_wait(t_sample)

                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    dpos = sm_state[:3] * (args.max_pos_speed / frequency)
                    drot_xyz = sm_state[3:] * (args.max_rot_speed / frequency)
                    drot = st.Rotation.from_euler("xyz", drot_xyz)

                    (
                        keyboard_action,
                        collect_enum,
                    ) = self.device_interface.get_action()  # from the keyboard

                    # If undo action is taken, undo the last 10 actions.
                    if collect_enum == CollectEnum.UNDO:
                        self.verbose_print("Undo the last 10 actions.")
                        self.obs = self.obs[:-10]

                        # Set the environment to the state before the last 10 actions.
                        self.env.reset_env_to(env_idx=0, state=self.obs[-1])
                        self.env.refresh()

                        translation, quat_xyzw = self.env.get_ee_pose()

                        translation, quat_xyzw = (
                            translation.cpu().numpy().squeeze(),
                            quat_xyzw.cpu().numpy().squeeze(),
                        )
                        gripper_width = self.env.gripper_width()
                        rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                        target_pose_rv = np.array([*translation, *rotvec])
                        gripper_open = gripper_width >= 0.06
                        grasp_flag = torch.from_numpy(
                            np.array([-1 if gripper_open else 1])
                        ).to(env_device)

                        target_pose_last_action_rv = None

                        continue

                    if np.allclose(dpos, 0.0) and np.allclose(drot_xyz, 0.0):
                        action_taken = False
                        if target_pose_last_action_rv is None:
                            translation, quat_xyzw = self.env.get_ee_pose()
                            translation, quat_xyzw = (
                                translation.cpu().numpy().squeeze(),
                                quat_xyzw.cpu().numpy().squeeze(),
                            )
                            rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
                            target_pose_last_action_rv = np.array(
                                [*translation, *rotvec]
                            )
                    else:
                        action_taken = True
                        target_pose_last_action_rv = None

                    steps_since_grasp += 1
                    if steps_since_grasp > 10:
                        ready_to_grasp = True
                    if steps_since_grasp < 10:
                        action_taken = True

                    kb_grasp = prev_keyboard_gripper != keyboard_action[-1]
                    sm_grasp = (
                        sm.is_button_pressed(0) or sm.is_button_pressed(1)
                    ) and ready_to_grasp
                    if kb_grasp or sm_grasp:
                        # env.gripper_close() if gripper_open else env.gripper_open()
                        grasp_flag = -1 * grasp_flag
                        gripper_open = not gripper_open

                        ready_to_grasp = False
                        steps_since_grasp = 0
                    prev_keyboard_gripper = keyboard_action[-1]

                    new_target_pose_rv = target_pose_rv.copy()
                    new_target_pose_rv[:3] += dpos
                    new_target_pose_rv[3:] = (
                        drot * st.Rotation.from_rotvec(target_pose_rv[3:])
                    ).as_rotvec()

                    target_pose_mat = pose_rv2mat(target_pose_rv)
                    if target_pose_last_action_rv is not None:
                        new_target_pose_mat = pose_rv2mat(target_pose_last_action_rv)
                    else:
                        new_target_pose_mat = pose_rv2mat(new_target_pose_rv)

                    # convert this into the furniture bench info we need
                    # action, collect_enum = to_isaac_pose(new_target_pose), CollectEnum.DONE_FALSE  # TODO
                    action = to_isaac_dpose_from_abs(
                        current_pose_mat=target_pose_mat,
                        goal_pose_mat=new_target_pose_mat,
                        grasp_flag=grasp_flag,
                        device=env_device,
                        rm=self.right_multiply_rot,
                    )

                    pos_bounds_m = 0.02 if self.env.ctrl_mode == "diffik" else 0.025
                    ori_bounds_deg = 15 if self.env.ctrl_mode == "diffik" else 20

                    # bp()
                    action = scale_scripted_action(
                        action.detach().cpu().clone(),
                        pos_bounds_m=pos_bounds_m,
                        ori_bounds_deg=ori_bounds_deg,
                        device=self.env.device,
                    )

                    if not (np.allclose(keyboard_action[:6], 0.0)):
                        action[0, :7] = (
                            torch.from_numpy(keyboard_action[:7])
                            .float()
                            .to(action.device)
                        )
                        action_taken = True
                        target_pose_last_action_rv = None

                    skill_complete = int(collect_enum == CollectEnum.SKILL)
                    if skill_complete == 1:
                        self.skill_set.append(skill_complete)

                    if collect_enum == CollectEnum.TERMINATE:
                        self.verbose_print("Terminate the program.")
                        break

                    # An episode is done.
                    if done or collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                        if self.is_sim:
                            # Convert it to numpy.
                            for k, v in next_obs.items():
                                if isinstance(v, dict):
                                    for k1, v1 in v.items():
                                        v[k1] = self._squeeze_and_numpy(v1)
                                else:
                                    next_obs[k] = self._squeeze_and_numpy(v)

                        self.org_obs.append(next_obs)

                        n_ob = {}
                        n_ob["color_image1"] = resize(next_obs["color_image1"])
                        n_ob["color_image2"] = resize_crop(next_obs["color_image2"])
                        n_ob["robot_state"] = next_obs["robot_state"]
                        n_ob["parts_poses"] = next_obs["parts_poses"]
                        self.obs.append(n_ob)

                        if (
                            done and not self.env.furnitures[0].all_assembled()
                        ) or collect_enum is CollectEnum.FAIL:
                            collect_enum = CollectEnum.FAIL
                            if self.save_failure:
                                self.verbose_print("Saving failure trajectory.")
                                obs = self.save_and_reset(collect_enum, {})
                            else:
                                self.verbose_print(
                                    "Failed to assemble the furniture, reset without saving."
                                )
                                obs = self.reset()
                            self.num_fail += 1
                        else:
                            if done:
                                collect_enum = CollectEnum.SUCCESS

                            obs = self.save_and_reset(collect_enum, {})
                            self.num_success += 1
                            self.update_pbar()

                        self.traj_counter += 1
                        self.verbose_print(
                            f"Success: {self.num_success}, Fail: {self.num_fail}"
                        )

                        done = False

                        steps_since_grasp = 0
                        ready_to_grasp = True
                        target_pose_last_action_rv = None

                        gripper_open = gripper_width >= 0.95
                        grasp_flag = torch.from_numpy(
                            np.array([-1 if gripper_open else 1])
                        ).to(env_device)

                        continue

                    # Execute action.
                    next_obs, rew, done, info = self.env.step(action)

                    if rew == 1:
                        self.last_reward_idx = len(self.acts)

                    # Label reward.
                    if collect_enum == CollectEnum.REWARD:
                        rew = self.env.furniture.manual_assemble_label(
                            self.device_interface.rew_key
                        )
                        if rew == 0:
                            # Correction the label.
                            self.rews[self.last_reward_idx] = 0
                            rew = 1

                    # Error handling.
                    if not info["obs_success"]:
                        self.verbose_print(
                            "Getting observation failed, save trajectory."
                        )
                        # Pop the last reward and action so that obs has length plus 1 then those of actions and rewards.
                        self.rews.pop()
                        self.acts.pop()
                        obs = self.save_and_reset(CollectEnum.FAIL, info)
                        continue

                    # Logging a step.
                    if action_taken:
                        self.step_counter += 1
                        print(
                            f"{[self.step_counter]} assembled: {self.env.furniture.assembled_set} num assembled: {len(self.env.furniture.assembled_set)} Skill: {len(self.skill_set)}. (Z-rot: {self.device_interface.rot_fraction:.1%})"
                        )

                        # Store a transition.
                        if info["action_success"]:
                            if self.is_sim:
                                for k, v in obs.items():
                                    if isinstance(v, dict):
                                        for k1, v1 in v.items():
                                            v[k1] = (
                                                v1.squeeze().cpu().numpy()
                                                if isinstance(v1, torch.Tensor)
                                                else v1
                                            )
                                    else:
                                        obs[k] = (
                                            v.squeeze().cpu().numpy()
                                            if isinstance(v, torch.Tensor)
                                            else v
                                        )
                                if isinstance(rew, torch.Tensor):
                                    rew = float(rew.squeeze().cpu())

                            self.org_obs.append(obs.copy())

                            ob = {}
                            if (not self.is_sim) or (not self.resize_sim_img):
                                # Resize for every real world images, or for sim didn't resize in simulation side.
                                ob["color_image1"] = resize(obs["color_image1"])
                                ob["color_image2"] = resize_crop(obs["color_image2"])
                            else:
                                ob["color_image1"] = obs["color_image1"]
                                ob["color_image2"] = obs["color_image2"]
                            ob["robot_state"] = obs["robot_state"]
                            ob["parts_poses"] = obs["parts_poses"]
                            self.obs.append(ob)

                            if self.is_sim:
                                if isinstance(action, torch.Tensor):
                                    action = action.squeeze().cpu().numpy()
                                else:
                                    action = action.squeeze()
                            self.acts.append(action)
                            self.rews.append(rew)
                            self.skills.append(skill_complete)

                            # Intrinsic rotation
                            translation, quat_xyzw = self.env.get_ee_pose()
                            translation, quat_xyzw = (
                                translation.cpu().numpy().squeeze(),
                                quat_xyzw.cpu().numpy().squeeze(),
                            )
                    obs = next_obs

                    # target_pose = new_target_pose
                    translation, quat_xyzw = self.env.get_ee_pose()
                    translation, quat_xyzw = (
                        translation.cpu().numpy().squeeze(),
                        quat_xyzw.cpu().numpy().squeeze(),
                    )
                    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()

                    target_pose_rv = np.array([*translation, *rotvec])

                    # SM wait
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                self.verbose_print(
                    f"Collected {self.traj_counter} / {self.num_demos} successful trajectories!"
                )

    def save_and_reset(self, collect_enum: CollectEnum, info):
        """Saves the collected data and reset the environment."""
        self.save(collect_enum, info)
        self.verbose_print(f"Saved {self.traj_counter} trajectories in this run.")
        return self.reset()

    def reset(self):
        obs = self.env.reset()
        self._reset_collector_buffer()

        state = self.load_state(state=None)

        self.verbose_print("Start collecting the data!")
        if not self.scripted:
            self.verbose_print("Press enter to start")
            while True:
                if input() == "":
                    break
            time.sleep(0.2)

        return state

    def _reset_collector_buffer(self):
        self.obs = []
        self.org_obs = []
        self.acts = []
        self.rews = []
        self.skills = []
        self.step_counter = 0
        self.last_reward_idx = -1
        self.skill_set = []

    def save(self, collect_enum: CollectEnum, info):
        print(f"Length of trajectory: {len(self.obs)}")

        # Save transitions with resized images.
        data = {}
        data["observations"] = self.obs
        data["actions"] = self.acts
        data["rewards"] = self.rews
        data["skills"] = self.skills
        data["success"] = True if collect_enum == CollectEnum.SUCCESS else False
        data["furniture"] = self.furniture

        if "error" in info:
            data["error_description"] = info["error"].value
            data["error"] = True
        else:
            data["error"] = False
            data["error_description"] = ""

        # Save data.
        demo_path = self.data_path / ("success" if data["success"] else "failure")
        demo_path.mkdir(parents=True, exist_ok=True)

        path = demo_path / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

        if self.compress_pickles:
            # Add the suffix .gz if we are compressing the pickle files
            path = path.with_suffix(".pkl.xz")

        pickle_data(data, path)

        print(f"Data saved at {path}")

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def update_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def __del__(self):
        del self.env

        if self.device_interface is not None:
            self.device_interface.close()
