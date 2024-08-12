# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from typing import Dict
from src.dataset.normalizer import LinearNormalizer
from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
from src.common.geometry import proprioceptive_quat_to_6d_rotation
import torch
import src.common.geometry as G


from ipdb import set_trace as bp

# Set the gym logger to not print to console
import gym


class FurnitureEnvRLWrapper:

    def __init__(
        self,
        env: FurnitureRLSimEnv,
        max_env_steps=300,
        ee_dof=10,
        chunk_size=1,
        reset_on_success=False,
        normalize_reward=False,
        reward_clip=5.0,
        device="cuda",
    ):
        # super(FurnitureEnvWrapper, self).__init__(env)
        self.env = env
        self.chunk_size: int = chunk_size
        self.reset_on_success = reset_on_success
        self.device = device
        self.normalizer = LinearNormalizer()

        self.action_space = gym.spaces.Box(-1, 1, shape=(chunk_size, ee_dof))

        # Define a new observation space of dim 16 + 35 for 6D proprioception
        self.observation_space = gym.spaces.Box(
            -float("inf"),
            float("inf"),
            shape=(16 + env.get_parts_poses().shape[-1] + 7,),
        )
        # Define the maximum number of steps in the environment
        self.max_env_steps = max_env_steps
        self.num_envs = self.env.num_envs

        self.reward_normalizer = (
            RunningMeanStdClip(shape=(1,), clip_value=reward_clip)
            if normalize_reward
            else None
        )

        self.no_rotation_or_gripper = torch.tensor(
            [[0, 0, 0, 1, -1]], device=device, dtype=torch.float32
        ).repeat(self.num_envs, 1)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def process_obs(self, obs: Dict[str, torch.Tensor]):
        # Robot state is [pos, ori_quat, pos_vel, ori_vel, gripper]
        robot_state = obs["robot_state"]

        # Parts poses is [pos, ori_quat] for each part
        parts_poses = obs["parts_poses"]

        # Make the robot state have 6D proprioception
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        robot_state = self.normalizer(robot_state, "robot_state", forward=True)
        parts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        nobs = torch.cat([robot_state, parts_poses], dim=-1)

        # Clamp the observation to be bounded to [-5, 5]
        nobs = torch.clamp(nobs, -5, 5)

        return nobs

    def process_action(self, action: torch.Tensor):
        """
        Done any desired processing to the action before
        it is passed to the environment.
        """
        return action

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.process_obs(obs)

    def _inner_step(self, action_chunk: torch.Tensor):
        total_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        dones = torch.zeros(
            action_chunk.shape[0], dtype=torch.bool, device=action_chunk.device
        )
        for i in range(self.chunk_size):
            # The dimensions of the action_chunk are (num_envs, chunk_size, action_dim)
            obs, reward, done, info = self.env.step(action_chunk[:, i, :])
            total_reward += reward.squeeze()
            dones = dones | done.squeeze()

        return obs, total_reward, dones, info

    def step(self, naction_chunk: torch.Tensor):
        assert naction_chunk.shape[-2:] == self.action_space.shape

        # First denormalize the action
        action_chunk = self.normalizer(naction_chunk, "action", forward=False)

        # Move the robot
        obs, reward, terminated, info = self._inner_step(action_chunk)

        # Check if any envs have reached the max number of steps
        truncated = self.env.env_steps >= self.max_env_steps

        done = terminated | truncated

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer(reward)

        obs = self.process_obs(obs)

        return obs, reward, done, info

    def increment_randomness(self):
        self.env.increment_randomness()

    @property
    def force_magnitude(self):
        return self.env.max_force_magnitude

    @property
    def torque_magnitude(self):
        return self.env.max_torque_magnitude


class RunningMeanStdClip:
    def __init__(self, epsilon=1e-4, shape=(), clip_value=10.0, device="cuda"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.clip_value = clip_value

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def __call__(self, x):
        self.update(x)
        x_normalized = x / torch.sqrt(self.var + 1e-8)
        return torch.clamp(x_normalized, -self.clip_value, self.clip_value)


class ResidualPolicyEnvWrapper:

    def __init__(
        self,
        env: FurnitureRLSimEnv,
        max_env_steps=300,
        ee_dof=10,
        task="oneleg",
        normalize_reward=False,
        add_relative_pose=False,
        reset_on_success=True,
        reset_on_failure=False,
        reward_clip=5.0,
        sample_perturbations=False,
        device="cuda",
    ):
        # super(FurnitureEnvWrapper, self).__init__(env)
        self.env = env
        self.task = task
        self.add_relative_pose = add_relative_pose
        self.reset_on_success = reset_on_success
        self.reset_on_failure = reset_on_failure
        self.device = device
        self.reward_normalizer = (
            RunningMeanStdClip(shape=(1,), clip_value=reward_clip)
            if normalize_reward
            else None
        )
        self.sample_perturbations = sample_perturbations

        self.env_success = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.bool
        )

        # Define a new action space of dim 3 (x, y, z)
        self.action_space = gym.spaces.Box(-1, 1, shape=(ee_dof,))

        # Define a new observation space of dim 14 + 35 in range [-inf, inf] for quat proprioception
        # and 16 + 35 for 6D proprioception
        self.observation_space = gym.spaces.Box(
            -float("inf"),
            float("inf"),
            shape=(16 + env.get_parts_poses().shape[-1] + 7,),
        )

        # Define the maximum number of steps in the environment
        self.max_env_steps = max_env_steps
        self.num_envs = self.env.num_envs

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.env_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        return obs

    def step(self, action: torch.Tensor):
        assert action.shape[1:] == self.action_space.shape

        # Move the robot
        obs, reward, termination, info = self.env.step(
            action, sample_perturbations=self.sample_perturbations
        )
        reward = reward.squeeze()
        termination = termination.squeeze()

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer(reward)

        truncation = self.env.env_steps >= self.max_env_steps

        # Clip the obs
        obs["robot_state"] = torch.clamp(obs["robot_state"], -3, 3)
        obs["parts_poses"] = torch.clamp(obs["parts_poses"], -3, 3)

        return obs, reward, termination, truncation, info

    def increment_randomness(self):
        self.env.increment_randomness()

    @property
    def force_magnitude(self):
        return self.env.max_force_magnitude

    @property
    def torque_magnitude(self):
        return self.env.max_torque_magnitude
