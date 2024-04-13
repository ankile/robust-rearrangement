# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from pathlib import Path
from typing import Dict
import furniture_bench  # noqa

import os
import random
import time
from dataclasses import dataclass
import math

from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.common.pytorch_util import dict_to_device
from src.dataset import get_normalizer
from src.dataset.dataloader import EndlessDataloader
from src.dataset.dataset import FurnitureStateDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange
import tyro
from torch.utils.tensorboard import SummaryWriter

from src.behavior.mlp import SmallAgent, ResidualMLPAgent, SmallAgentSimple

# Set the gym logger to not print to console
import gym

gym.logger.set_level(40)


from src.gym import get_env

from wandb import Api

api = Api()


def get_agent(
    run_id: str,
    device: torch.device,
    load_weights: bool = True,
    control_mode_override: str = None,
):
    # Load a pre-trained model from WandB
    run = api.run(run_id)
    # run = api.run("ankile/one_leg-mlp-state-1/runs/lq0c1oz4")

    cfg = run.config

    cfg: DictConfig = OmegaConf.create(
        cfg,
        flags={"readonly": True},
    )

    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(
        root=f"./models/{run.name}", exist_ok=True, replace=True
    ).name

    print(f"Model path: {model_path}")

    # Get the normalizer
    normalizer_type = cfg.get("data", {}).get("normalization", "min_max")
    normalizer = get_normalizer(
        normalizer_type=normalizer_type,
        control_mode=(
            cfg.control.control_mode
            if control_mode_override is None
            else control_mode_override
        ),
    )

    # TODO: Fix this properly, but for now have an ugly escape hatch
    # vision_encoder_field_hotfix(run, config)

    print(OmegaConf.to_yaml(cfg))

    # Make the actor
    agent = ResidualMLPAgent(device, normalizer, cfg)

    print("NBNB: This is a hack to load the model weights, please fix soon")
    # TODO: Fix this properly, but for now have an ugly escape hatch
    import torch.nn as nn

    agent.normalizer.stats["parts_poses"] = nn.ParameterDict(
        {
            "min": nn.Parameter(torch.zeros(35)),
            "max": nn.Parameter(torch.ones(35)),
        }
    )

    state_dict = torch.load(model_path)

    if not load_weights:
        state_dict = {k: v for k, v in state_dict.items() if "normalizer" in k}
        state_dict = {k: v for k, v in state_dict.items() if "action" not in k}

    # Load the model weights
    agent.load_state_dict(state_dict, strict=False)
    agent.normalizer._turn_off_gradients()
    agent.cuda()

    return agent


class NormalizeReward(gym.core.Wrapper):
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
        obs, rews, done, infos = self.env.step(action)

        self.returns = self.returns * self.gamma * (1 - done.float()) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, done, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / torch.sqrt(self.return_rms.var + self.epsilon)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    load_weights: bool = True
    """whether to load the weights of the model"""
    headless: bool = True
    """if toggled, the environment will be set to headless mode"""
    agent: str = "small"
    """the agent to use"""
    normalize_reward: bool = False
    """if toggled, the rewards will be normalized"""

    # Algorithm specific arguments
    # env_id: str = "HalfCheetah-v4"
    # """the id of the environment"""
    total_timesteps: int = 100_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-6
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_env_steps: int = 750
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    bc_coef: float = 0.0
    """coefficient of the behavior cloning loss"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    penalize_failures: bool = False
    """if toggled, failed episodes will be penalized"""
    simplified_task: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_steps: int = 0
    """the number of steps (computed in runtime)"""


def get_demo_data_loader(cfg, normalizer, batch_size, num_workers=4):
    demo_data = FurnitureStateDataset(
        dataset_paths=Path(cfg.data_path[0]),
        pred_horizon=cfg.data.pred_horizon,
        obs_horizon=cfg.data.obs_horizon,
        action_horizon=cfg.data.action_horizon,
        normalizer=normalizer,
        data_subset=cfg.data.data_subset,
        control_mode=cfg.control.control_mode,
        first_action_idx=cfg.actor.first_action_index,
        pad_after=cfg.data.get("pad_after", True),
        max_episode_count=cfg.data.get("max_episode_count", None),
    )

    demo_data_loader = EndlessDataloader(
        dataset=demo_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    return demo_data_loader


class ActionChunkWrapper:
    def __init__(self, env: FurnitureSimEnv, chunk_size: int):
        self.env = env
        self.chunk_size = chunk_size
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action_chunk):
        total_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        for i in range(self.chunk_size):
            # The dimensions of the action_chunk are (num_envs, chunk_size, action_dim)
            # bp()
            obs, reward, done, info = self.env.step(action_chunk[:, i, :])
            total_reward += reward.squeeze()
            if done.all():
                break
        return obs, total_reward, done, info


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device="cuda"):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.float64, device=device)

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


class FurnitureEnvWrapper:
    def __init__(self, env: FurnitureSimEnv, max_env_steps=300):
        # super(FurnitureEnvWrapper, self).__init__(env)
        self.env = env

        # Define a new action space of dim 3 (x, y, z)
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,))

        # Define a new observation space of dim 14 + 35 in range [-inf, inf]
        self.observation_space = gym.spaces.Box(
            -float("inf"), float("inf"), shape=(14 + 35,)
        )

        self.timestep = 0
        self.max_env_steps = max_env_steps
        self.num_envs = self.env.num_envs

        self.no_action = torch.tensor(
            [[0, 0, 0, 1, 0]], device=device, dtype=torch.float32
        ).repeat(self.num_envs, 1)

    def process_obs(self, obs: Dict[str, torch.Tensor]):
        robot_state = obs["robot_state"]
        parts_poses = obs["parts_poses"]

        obs = torch.cat([robot_state, parts_poses], dim=-1)
        return obs

    def reset(self, **kwargs):
        self.timestep = 0
        obs = self.env.reset()
        return self.process_obs(obs)

    def step(self, action: np.ndarray):
        # Accept delta actions in range [-1, 1] -> normalize and clip to [-0.025, 0.025]
        # bp()
        action = torch.clamp(action, -1, 1) * 0.025

        # Accept actions of dim 3 (x, y, z) and convert to dim 6 (x, y, z, qx=0, qy=0, qz=0, qw=1, gripper=0)
        action = torch.cat(
            [
                action,
                self.no_action,
            ],
            dim=-1,
        )

        # Move the robot
        obs, reward, done, info = self.env.step(action)

        # Calculate reward
        # Get the end effector position
        ee_pos = obs["robot_state"][:, :3]

        # Make a dense reward that measures the distance to the goal: [ 0.5934, -0.2813,  0.5098]
        goal = torch.tensor([0.5934, -0.2813, 0.5098], device=device)

        # Set the reward to be 1 if the distance is less than 0.1 (10 cm) from the goal
        reward[torch.norm(ee_pos - goal, dim=-1) < 0.10] = 1
        reward = reward.squeeze()
        done = done.squeeze()

        # bp()
        obs = self.process_obs(obs)

        # Check if the episode is done
        self.timestep += 1
        if episode_ended := self.timestep >= self.max_env_steps:
            obs = self.reset()

        done = torch.full_like(done, episode_ended)
        # truncated = np.full_like(done, False)

        # return obs, reward, terminated, truncated, info
        return obs, reward, done, info


if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"one_leg__{args.exp_name}__chunked__bc_coef_{args.bc_coef}__delta__simplified_{args.simplified_task}__{args.agent}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/debug2/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda")
    act_rot_repr = "quat"  # "quat" or "rot_6d"
    action_type = "delta"  # "delta" or "pos"
    headless = True
    num_env_steps = 300
    num_envs = args.num_envs

    # env setup
    env: FurnitureSimEnv = get_env(
        act_rot_repr=act_rot_repr,
        action_type=action_type,
        april_tags=False,
        ctrl_mode="diffik",
        furniture="one_leg",
        gpu_id=0,
        headless=headless,
        num_envs=num_envs,
        observation_space="state",
        randomness="low",
        pos_scalar=1,
        rot_scalar=1,
        stiffness=1000,
        damping=200,
    )
    env = FurnitureEnvWrapper(env, max_env_steps=num_env_steps)

    if args.normalize_reward:
        env = NormalizeReward(env, device=device)

    # assert isinstance(
    #     envs.single_action_space, gym.spaces.Box
    # ), "only continuous action space is supported"
    from ipdb import set_trace as bp

    agent = SmallAgentSimple(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    args.num_steps = math.ceil(args.num_env_steps / agent.action_horizon)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(
        f"With chunk size {agent.action_horizon}, we have {args.num_steps} policy steps."
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + env.observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + env.action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # # Get the dataloader for the demo data for the behavior cloning
    # demo_data_loader = get_demo_data_loader(
    #     cfg=agent.config,
    #     normalizer=agent.normalizer.get_copy().cpu(),
    #     batch_size=args.minibatch_size,
    # )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # bp()
    next_done = torch.zeros(args.num_envs).to(device)
    next_obs = env.reset()

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iteration: {iteration}/{args.num_iterations}")

        # Reward normalization
        rew_normalizer = RunningMeanStd(shape=())
        running_returns = torch.zeros(
            (args.num_envs, 1), device=device, dtype=torch.float32
        )

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # bp()
            with torch.no_grad():
                # bp()
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, infos = env.step(action)

            # Get the end effector position
            # ee_pos, _ = env.env.get_ee_pose()

            # Make a dense reward that measures the distance to the goal: [ 0.5934, -0.2813,  0.5098]
            # goal = torch.tensor([0.5934, -0.2813, 0.5098], device=device)
            # reward = 1 / (1 + torch.norm(ee_pos - goal, dim=-1))

            # Set the reward to be 1 if the distance is less than 0.1 (10 cm) from the goal
            # reward[torch.norm(ee_pos - goal, dim=-1) < 0.10] = 1
            reward = reward.view(-1)

            # Normalize the reward
            # running_returns = 0.99 * running_returns + reward
            # rew_normalizer.update(running_returns)
            # # bp()
            # reward = reward / torch.sqrt(rew_normalizer.var + 1e-8)
            # reward = torch.clamp(reward, -5, 5)

            rewards[step] = reward
            next_done = next_done.view(-1)

            if (env_step := step * agent.action_horizon + 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()}"
                )

        # Calculate the discounted rewards
        discounted_rewards = (
            (
                rewards
                * args.gamma
                ** torch.arange(args.num_steps, device=device).float().unsqueeze(1)
            )
            .sum(dim=0)
            .mean()
            .item()
        )
        # If any positive reward was received, consider it a success
        success_rate = (rewards > 0).any(dim=0).float().mean().item()

        print(
            f"Mean episode return: {discounted_rewards}, Success rate: {success_rate}"
        )

        # bootstrap value if not done
        # NOTE: Solve resetting of the environment correctly
        # bp()

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.to(torch.float)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
                    nextvalues = values[t + 1]

                # bp()
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # bp()

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # demo_data_iter = iter(demo_data_loader)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in trange(args.update_epochs, desc="Policy update"):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                # bp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Clip the value loss
                v_loss = torch.clamp(v_loss, -1, 1)
                loss = v_loss * args.vf_coef

                # Entropy loss
                entropy_loss = entropy.mean()

                loss += pg_loss - args.ent_coef * entropy_loss

                # # Behavior cloning loss
                # batch = next(demo_data_iter)
                # batch = dict_to_device(batch, device)

                # # Get loss
                # bc_loss = agent.compute_loss(batch)

                # loss += args.bc_coef * bc_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/bc_loss", bc_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("charts/rewards", rewards.sum().item(), global_step)
        writer.add_scalar("charts/discounted_rewards", discounted_rewards, global_step)
        writer.add_scalar("charts/success_rate", success_rate, global_step)

        # Add histograms
        writer.add_histogram("histograms/values", values, global_step)
        writer.add_histogram("histograms/returns", returns, global_step)
        writer.add_histogram("histograms/advantages", advantages, global_step)
        writer.add_histogram("histograms/logprobs", logprobs, global_step)
        writer.add_histogram("histograms/rewards", rewards, global_step)

        # Add histograms for the actions
        writer.add_histogram("actions/x", actions[:, :, 0], global_step)
        writer.add_histogram("actions/y", actions[:, :, 1], global_step)
        writer.add_histogram("actions/z", actions[:, :, 2], global_step)

        # Add histograms for the gradients and the weights
        for name, param in agent.named_parameters():
            writer.add_histogram(f"weights/{name}", param, global_step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, global_step)

    print(f"Training finished in {(time.time() - start_time):.2f}s")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    writer.close()
