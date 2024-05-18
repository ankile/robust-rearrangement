from src.gym.furniture_sim_env import FurnitureRLSimEnv
import torch

from ipdb import set_trace as bp

# Set the gym logger to not print to console
import gym


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device="cuda"):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=device)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_normalize(self, x):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.update(x)
        self.update(self.returns)
        return x / torch.sqrt(self.var + self.epsilon)


class NormalizeObservation(gym.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.Wrapper.__init__(self, env)

        self.num_envs = env.num_envs
        self.is_vector_env = True

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape, device="cuda")
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, done, truncated, infos = self.env.step(action)
        obs = self.normalize(obs)
        return obs, rews, done, truncated, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs = self.env.reset(**kwargs)

        return self.normalize(obs)

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeRewardWrapper(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        device: str = "cpu",
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
            device (str): The device to use for PyTorch tensors ("cpu" or "cuda").
        """
        gym.Wrapper.__init__(self, env)

        self.num_envs = env.num_envs
        self.is_vector_env = True

        self.return_rms = RunningMeanStd(shape=(), device=device)
        self.returns = torch.zeros(self.num_envs, device=device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, done, truncated, infos = self.env.step(action)

        self.returns = self.returns * self.gamma * (1 - done.float()) + rews
        rews = self.normalize(rews)

        return obs, rews, done, truncated, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / torch.sqrt(self.return_rms.var + self.epsilon)


class ActionChunkWrapper:
    def __init__(self, env: FurnitureRLSimEnv, chunk_size: int):
        self.env = env
        self.chunk_size = chunk_size

        # Update the action space to have the correct shape, i.e., insert the chunk size
        self.env.action_space = gym.spaces.Box(
            low=env.action_space.low.repeat(chunk_size, axis=0),
            high=env.action_space.high.repeat(chunk_size, axis=0),
        )

    def reset(self, env_idxs: torch.tensor = None):
        return self.env.reset(env_idxs=env_idxs)

    def step(self, action_chunk):
        total_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        dones = torch.zeros(
            action_chunk.shape[0], dtype=torch.bool, device=action_chunk.device
        )
        for i in range(self.chunk_size):
            # The dimensions of the action_chunk are (num_envs, chunk_size, action_dim)
            # bp()
            obs, reward, done, info = self.env.step(action_chunk[:, i, :])
            total_reward += reward.squeeze()
            dones = dones | done.squeeze()

        return obs, total_reward, done, info
