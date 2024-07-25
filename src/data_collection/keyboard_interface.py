"""Reference: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/devices/keyboard.py"""

import gym
import numpy as np
from pynput.keyboard import Key, Listener

from furniture_bench.device.device_interface import DeviceInterface
from src.data_collection.collect_enum import CollectEnum
import furniture_bench.utils.transform as T


class KeyboardInterface(DeviceInterface):
    """Define keyboard interface to control franka."""

    POSE_ACTIONS = ["s", "w", "a", "d", "e", "q"]
    GRIP_ACTIONS = ["z"]
    ROT_ACTIONS = ["i", "k", "j", "l", "u", "o"]

    # Only these actions are exposed to gym environment.
    ACTIONS = POSE_ACTIONS + GRIP_ACTIONS + ROT_ACTIONS

    ADJUST_DELTA = ["[", "]"]

    # INIT_POS_DELTA = 0.02
    INIT_POS_DELTA = 0.01
    INIT_ROT_DELTA = 0.2  # Radian.

    MAX_POS_DELTA = 0.1
    MAX_ROT_DELTA = 0.2  # Radian.

    intr_z_limit = (-15, 17)

    def __init__(self):
        self.reset()

        # Make a thread to listen to keyboard and register callback functions.
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def reset(self):
        self.pos_delta = KeyboardInterface.INIT_POS_DELTA
        self.rot_delta = KeyboardInterface.INIT_ROT_DELTA
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = self.pos.copy()
        self.ori = np.zeros(3)  # (Roll, Pitch, Yaw)
        self.last_ori = self.ori.copy()
        self.grasp = np.array([-1])
        self.rew_key = 0

        self.key_enum = CollectEnum.DONE_FALSE

    def on_press(self, k):
        try:
            k = k.char

            # Moving arm.
            if k in KeyboardInterface.ACTIONS:
                if k in KeyboardInterface.POSE_ACTIONS:
                    self._pose_action(k)
                elif k in KeyboardInterface.GRIP_ACTIONS:
                    self._grip_action(k)
                elif k in KeyboardInterface.ROT_ACTIONS:
                    self._rot_action(k)

            # Data labelling and debugging.
            elif k in KeyboardInterface.ADJUST_DELTA:
                self._adjust_delta(k)
            elif k == "t":
                self.key_enum = CollectEnum.SUCCESS
            elif k == "y":
                self.key_enum = CollectEnum.SUCCESS_RECORD
            elif k == "n":
                self.key_enum = CollectEnum.FAIL
            elif k.isdigit():
                gym.logger.info(f"Reward pressed: {k}")
                self.rew_key = int(k)
                self.key_enum = CollectEnum.REWARD
            elif k == "`":
                gym.logger.info("Skill complete pressed")
                self.key_enum = CollectEnum.SKILL
            elif k == "r":
                gym.logger.info("Reset pressed")
                self.key_enum = CollectEnum.RESET
            elif k == "p":
                gym.logger.info("Pause pressed")
                self.key_enum = CollectEnum.PAUSE
            elif k == "c":
                gym.logger.info("Continue pressed")
                self.key_enum = CollectEnum.CONTINUE
            # Add undo action.
            elif k == "b":
                gym.logger.info("Undo pressed")
                self.key_enum = CollectEnum.UNDO
        except AttributeError as e:
            pass

    def on_release(self, k):
        try:
            # Terminates keyboard monitoring.
            if k == Key.esc:
                return False
        except AttributeError as e:
            pass

    def _pose_action(self, k):
        if k == "w":
            self.pos[0] -= self.pos_delta
        elif k == "s":
            self.pos[0] += self.pos_delta
        elif k == "a":
            self.pos[1] -= self.pos_delta
        elif k == "d":
            self.pos[1] += self.pos_delta
        elif k == "q":
            self.pos[2] -= self.pos_delta
        elif k == "e":
            self.pos[2] += self.pos_delta

    def _grip_action(self, k):
        if k == "z":
            self.grasp = -self.grasp

    def _rot_action(self, k):
        if k == "k":
            self.ori[1] += self.rot_delta
        elif k == "i":
            self.ori[1] -= self.rot_delta
        elif k == "j":
            self.ori[0] += self.rot_delta
        elif k == "l":
            self.ori[0] -= self.rot_delta
        elif k == "o":
            self.ori[2] -= self.rot_delta
        elif k == "u":
            self.ori[2] += self.rot_delta

    @property
    def rot_fraction(self):
        return (self.ori[2] - self.intr_z_limit[0]) / (
            self.intr_z_limit[1] - self.intr_z_limit[0]
        )

    def _adjust_delta(self, k):
        if k == "]":
            # Use larger step size of movement.
            self.pos_delta += 0.001
            self.rot_delta += 0.05
        elif k == "[":
            # Use smaller step size of movement.
            self.pos_delta -= 0.001
            self.rot_delta -= 0.05
            # Prevent becomming negative value.
            self.pos_delta = min(self.pos_delta, KeyboardInterface.MAX_POS_DELTA)
            self.rot_delta = min(self.rot_delta, KeyboardInterface.MAX_ROT_DELTA)
        gym.logger.info(
            "pose delta: {:.3f}, rotation delta: {:.3f}".format(
                self.pos_delta, self.rot_delta
            )
        )

    def get_action(self, use_quat=True):
        dpos = self.pos - self.last_pos
        dori = self.ori - self.last_ori

        self.last_pos = self.pos.copy()
        self.last_ori = self.ori.copy()

        if use_quat:
            dquat = T.mat2quat(T.euler2mat(dori))
            # Use positive element for the first element of quaternion (ease of learning).
            ret = np.concatenate([dpos, dquat, self.grasp]), self.key_enum
        else:
            ret = np.concatenate([dpos, dori, self.grasp]), self.key_enum
        self.key_enum = CollectEnum.DONE_FALSE
        return ret

    def print_usage(self):
        print("==============Keyboard Usage=================")
        print("Positional movements in base frame")
        print("q (- z-axis) w (- x-axis) e (+ z-axis)")
        print("a (- y-axis) s (+ x-axis) d (+ y-axis)")

        print("Rotational movements in base frame")
        print("u (- z-axis-rot) i (neg y-axis-rot)  o (+ z-axis)")
        print("j (pos x-axis-rot) k (pos y-axis-rot)  l (neg x-axis-rot)")

        print("Toggle gripper open and close")
        print("z")
        print("===============================")

    def close(self):
        self.listener.stop()
