"""Define data collection class that rollout the environment, get action from the interface (e.g., teleoperation, automatic scripts), and save data."""
import lzma
import time
import pickle
from datetime import datetime
from pathlib import Path
import gzip

import gym
import torch

from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.sim_config import sim_config
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

from src.data_processing.utils import resize, resize_crop
from src.visualization.render_mp4 import pickle_data


class DataCollector:
    """Demonstration collection class.

    Only used for collecting scripted demos in the simulator

    For teleoperation demos, use `DataCollectorSM` instead.

    For now, only stores pickle files, either compressed or uncompressed.
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
        save_failure: bool = False,
        num_demos: int = 100,
        resize_sim_img: bool = False,
        ctrl_mode: str = "osc",
        compress_pickles: bool = False,
        verbose: bool = False,
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
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            pkl_only (bool): Whether to save only `pkl` files (i.e., exclude *.mp4 and *.png).
            save_failure (bool): Whether to save failure trajectories.
            num_demos (int): The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
            resize_sim_img (bool): Read resized image
            ctrl_mode (str): 'osc' (joint torque, with operation space control) or 'diffik' (joint impedance, with differential inverse kinematics control)
            compress_pickles (bool): Whether to compress the pickle files with gzip.
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

        self.save_failure = save_failure
        self.resize_sim_img = resize_sim_img
        self.compress_pickles = compress_pickles
        self.verbose = verbose

        self._reset_collector_buffer()

    def _verbose_print(self, msg):
        if self.verbose:
            print(msg)

    def collect(self):
        print("[data collection] Start collecting the data!")

        obs = self.reset()
        done = False

        while self.num_success < self.num_demos:
            # Get an action.
            if self.scripted:
                action, skill_complete = self.env.get_assembly_action()
                pos_bounds_m = 0.02 if self.env.ctrl_mode == "diffik" else 0.025
                ori_bounds_deg = 15 if self.env.ctrl_mode == "diffik" else 20
                action = scale_scripted_action(
                    action.detach().cpu().clone(),
                    pos_bounds_m=pos_bounds_m,
                    ori_bounds_deg=ori_bounds_deg,
                    device=self.env.device,
                )
                collect_enum = CollectEnum.DONE_FALSE
            else:
                action, collect_enum = self.device_interface.get_action()
                skill_complete = int(collect_enum == CollectEnum.SKILL)
                if skill_complete == 1:
                    self.skill_set.append(skill_complete)

            if collect_enum == CollectEnum.TERMINATE:
                print("Terminate the program.")
                break

            # An episode is done.
            if done or collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                if self.is_sim:
                    # Convert it to numpy.
                    for k, v in next_obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = v1.squeeze().cpu().numpy()
                        elif k == "color_image1":
                            next_obs[k] = resize(next_obs[k]).squeeze().cpu().numpy()
                        elif k == "color_image2":
                            next_obs[k] = (
                                resize_crop(next_obs[k]).squeeze().cpu().numpy()
                            )
                        else:
                            next_obs[k] = v.squeeze().cpu().numpy()

                self.org_obs.append(next_obs)

                n_ob = {}
                n_ob["color_image1"] = next_obs["color_image1"]
                n_ob["color_image2"] = next_obs["color_image2"]
                n_ob["robot_state"] = next_obs["robot_state"]
                n_ob["parts_poses"] = next_obs["parts_poses"]
                self.obs.append(n_ob)

                if done and not self.env.furnitures[0].all_assembled():
                    if self.save_failure:
                        print("Saving failure trajectory.")
                        collect_enum = CollectEnum.FAIL
                        obs = self.save_and_reset(collect_enum, {})
                    else:
                        print("Failed to assemble the furniture, reset without saving.")
                        obs = self.reset()
                        collect_enum = CollectEnum.SUCCESS
                    self.num_fail += 1
                else:
                    if done:
                        collect_enum = CollectEnum.SUCCESS

                    obs = self.save_and_reset(collect_enum, {})
                    self.num_success += 1
                self.traj_counter += 1
                print(f"Success: {self.num_success}, Fail: {self.num_fail}")
                done = False
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
                print("Getting observation failed, save trajectory.")
                # Pop the last reward and action so that obs has length plus 1 then those of actions and rewards.
                self.rews.pop()
                self.acts.pop()
                obs = self.save_and_reset(CollectEnum.FAIL, info)
                continue

            # Logging a step.
            self.step_counter += 1
            self._verbose_print(
                f"{[self.step_counter]} assembled: {self.env.furniture.assembled_set} num assembled: {len(self.env.furniture.assembled_set)} Skill: {len(self.skill_set)}"
            )

            # Store a transition.
            if info["action_success"]:
                if self.is_sim:
                    for k, v in obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = v1.squeeze().cpu().numpy()
                        elif k == "color_image1":
                            obs[k] = resize(obs[k]).squeeze().cpu().numpy()
                        elif k == "color_image2":
                            obs[k] = resize_crop(obs[k]).squeeze().cpu().numpy()
                        else:
                            obs[k] = v.squeeze().cpu().numpy()
                    if isinstance(rew, torch.Tensor):
                        rew = float(rew.squeeze().cpu())

                self.org_obs.append(obs.copy())
                ob = {}
                # if (not self.is_sim) or (not self.resize_sim_img):
                #     # Resize for every real world images, or for sim didn't resize in simulation side.
                #     ob["color_image1"] = resize(obs["color_image1"])
                #     ob["color_image2"] = resize_crop(obs["color_image2"])
                # else:
                #     ob["color_image1"] = obs["color_image1"]
                #     ob["color_image2"] = obs["color_image2"]
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
            obs = next_obs

        print(
            f"Collected {self.traj_counter} / {self.num_demos} successful trajectories!"
        )

    def save_and_reset(self, collect_enum: CollectEnum, info):
        """Saves the collected data and reset the environment."""
        self.save(collect_enum, info)
        print(f"Saved {self.traj_counter} trajectories in this run.")
        return self.reset()

    def reset(self):
        obs = self.env.reset()
        self._reset_collector_buffer()

        print("Start collecting the data!")
        if not self.scripted:
            print("Press enter to start")
            while True:
                if input() == "":
                    break
            time.sleep(0.2)

        return obs

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

    def __del__(self):
        del self.env

        if self.device_interface is not None:
            self.device_interface.close()
