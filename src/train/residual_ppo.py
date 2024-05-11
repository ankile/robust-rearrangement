from collections import deque
from typing import Dict, Literal, Tuple
import furniture_bench  # noqa

import random
import time
from dataclasses import dataclass
import math

from src.behavior import get_actor
from src.behavior.base import Actor
from src.common.files import get_processed_paths
from src.eval.load_model import load_bc_actor
from src.gym.env_rl_wrapper import ResidualPolicyEnvWrapper
from src.gym.furniture_sim_env import (
    FurnitureRLSimEnv,
    FurnitureRLSimEnvPlaceTabletop,
    FurnitureRLSimEnvReacher,
)
from furniture_bench.envs.observation import DEFAULT_STATE_OBS
import numpy as np
from src.common.context import suppress_all_output
from src.dataset.dataset import FurnitureStateDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import tyro
from torch.utils.tensorboard import SummaryWriter


from src.behavior.mlp import (
    SmallMLPAgent,
)


from ipdb import set_trace as bp

# Set the gym logger to not print to console
import gym

gym.logger.set_level(40)
import wandb


from src.gym import turn_off_april_tags


@dataclass
class Args:
    exp_name: Literal[
        "reacher",
        "oneleg",
        "place-tabletop",
    ] = "oneleg"
    """the name of this experiment"""
    seed: int = None
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
    headless: bool = True
    """if toggled, the environment will be set to headless mode"""
    agent: Literal[
        "small",
        "big",
        "residual",
        "residual-separate",
    ] = "small"
    """the agent to use"""
    normalize_reward: bool = False
    """if toggled, the rewards will be normalized"""
    normalize_obs: bool = False
    """if toggled, the observations will be normalized"""
    recalculate_advantages: bool = False
    """if toggled, the advantages will be recalculated every update epoch"""
    init_logstd: float = 0.0
    """the initial value of the log standard deviation"""
    ee_dof: int = 3
    """the number of degrees of freedom of the end effector"""
    bc_loss_type: Literal["mse", "nll"] = "mse"
    """the type of the behavior cloning loss"""
    supervise_value_function: bool = False
    """if toggled, the value function will be supervised"""
    action_type: Literal["delta", "pos"] = "delta"
    """the type of the action space"""
    chunk_size: int = 1
    """the chunk size for the action space"""
    data_collection_steps: int = None
    """the number of steps to collect data"""
    add_relative_pose: bool = False
    """if toggled, the relative pose will be added to the observation"""
    n_iterations_train_only_value: int = 0
    """the number of iterations to train only the value function"""
    load_checkpoint: bool = True
    """the checkpoint to load the model from"""
    reset_on_failure: bool = False
    """if toggled, the environment will be reset on failure"""
    debug: bool = False
    """if toggled, the debug mode will be enabled"""

    # Algorithm specific arguments
    total_timesteps: int = 40_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_env_steps: int = 300
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 128
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
    residual_regularization: float = 0.0
    """the regularization coefficient for the residual policy"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_steps: int = 0
    """the number of steps (computed in runtime)"""
    continue_run_id: str = None
    """the run id to continue training from"""


def get_demo_data_loader(
    control_mode,
    n_batches,
    task,
    act_rot_repr="quat",
    num_workers=4,
    normalizer=None,
    action_horizon=1,
    add_relative_pose=False,
) -> DataLoader:

    paths = get_processed_paths(
        controller="diffik",
        domain="sim",
        task="one_leg",
        demo_source="teleop",
        randomness=["low", "med"],
        demo_outcome="success",
    )

    demo_data = FurnitureStateDataset(
        dataset_paths=paths,
        obs_horizon=1,
        pred_horizon=action_horizon,
        action_horizon=action_horizon,
        normalizer=normalizer,
        data_subset=None,
        control_mode=control_mode,
        act_rot_repr=act_rot_repr,
        first_action_idx=0,
        pad_after=False,
        max_episode_count=None,
        task=task,
        add_relative_pose=add_relative_pose,
    )

    batch_size = len(demo_data) // n_batches

    demo_data_loader = DataLoader(
        dataset=demo_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )

    return demo_data_loader


@torch.no_grad()
def calculate_advantage(
    args: Args,
    device: torch.device,
    agent: SmallMLPAgent,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
):
    # bp()
    values = agent.get_value(obs).squeeze()
    next_value = agent.get_value(next_obs).reshape(1, -1)

    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - dones[-1].to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


def get_dataset_action(dataset, step, episode):
    ep_ends = dataset.episode_ends
    start_idx = ep_ends[episode - 1] if episode > 0 else 0
    action = dataset[start_idx + step]["action"]
    action = action.to(device)
    return action


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


from torch.distributions.normal import Normal


class ResidualPolicy(nn.Module):
    def __init__(self, obs_shape, action_shape, init_logstd=0):
        """
        Args:
            obs_shape: the shape of the observation (i.e., state + base action)
            action_shape: the shape of the action (i.e., residual, same size as base action)
        """
        super().__init__()

        self.action_dim = action_shape[-1]
        self.obs_dim = np.prod(obs_shape) + np.prod(action_shape)

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, np.prod(action_shape), bias=False), std=0.1),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * init_logstd)

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)

    def get_action_and_value(
        self, nobs: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean: torch.Tensor = self.actor_mean(nobs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(dim=1),
            probs.entropy().sum(dim=1),
            self.critic(nobs),
            action_mean,
        )


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.continue_run_id = None

    # TRY NOT TO MODIFY: seeding
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    if args.continue_run_id is not None:
        run_name = args.continue_run_id
    else:
        run_name = f"{int(time.time())}__{args.exp_name}__{args.agent}__{args.seed}"

    if args.data_collection_steps is None:
        args.data_collection_steps = args.num_env_steps

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

    run_directory = f"runs/debug-{args.exp_name}-residual-8"
    run_directory += "-delete" if args.debug else ""
    print(f"Run directory: {run_directory}")
    writer = SummaryWriter(f"{run_directory}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda")
    act_rot_repr = "rot_6d" if args.ee_dof == 10 else "quat"
    action_type = args.action_type

    turn_off_april_tags()

    # env setup
    with suppress_all_output(False):
        kwargs = dict(
            act_rot_repr=act_rot_repr,
            action_type=action_type,
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture="one_leg",
            gpu_id=0,
            headless=args.headless,
            num_envs=args.num_envs,
            observation_space="state",
            randomness="low",
            max_env_steps=100_000_000,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        if args.exp_name == "oneleg":
            env: FurnitureRLSimEnv = FurnitureRLSimEnv(**kwargs)
            # env: FurnitureSimEnv = FurnitureSimEnv(**kwargs)
        elif args.exp_name == "place-tabletop":
            env: FurnitureRLSimEnv = FurnitureRLSimEnvPlaceTabletop(**kwargs)
        elif args.exp_name == "reacher":
            assert args.bc_coef == 0, "Behavior cloning is not supported for Reacher"
            env: FurnitureRLSimEnv = FurnitureRLSimEnvReacher(**kwargs)
        else:
            raise ValueError(f"Unknown experiment name: {args.exp_name}")

    env.max_force_magnitude = 0.05
    env.max_torque_magnitude = 0.0025

    # Load the behavior cloning actor
    # TODO: The actor should keep tack of its own deque of observations
    # Similar to how it keeps track of a deque of actions
    # bc_actor: Actor = load_bc_actor("ankile/one_leg-mlp-state-1/runs/1ghcw9lu")
    bc_actor: Actor = load_bc_actor("ankile/one_leg-diffusion-state-1/runs/7623y5vn")

    env: ResidualPolicyEnvWrapper = ResidualPolicyEnvWrapper(
        env,
        max_env_steps=args.num_env_steps,
        ee_dof=args.ee_dof,
        task=args.exp_name,
        add_relative_pose=args.add_relative_pose,
        reset_on_failure=args.reset_on_failure,
    )
    env.set_normalizer(bc_actor.normalizer)

    residual_policy = ResidualPolicy(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        init_logstd=args.init_logstd,
    )
    residual_policy.to(device)

    optimizer = optim.AdamW(
        residual_policy.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=1e-6,
    )

    policy_steps = math.ceil(args.num_env_steps / 1)
    args.num_steps = args.data_collection_steps
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # print(
    #     f"With chunk size {agent.action_horizon}, we have {policy_steps} policy steps."
    # )
    print(f"Total timesteps: {args.total_timesteps}, batch size: {args.batch_size}")
    print(
        f"Mini-batch size: {args.minibatch_size}, num iterations: {args.num_iterations}"
    )

    # ALGO Logic: Storage setup
    best_success_rate = 0.1  # only save models that are better than this
    best_mean_episode_return = -1000
    running_mean_success_rate = 0.0
    decrease_lr_counter = 0
    steps_since_last_randomness_increase = 0

    obs = torch.zeros((args.num_steps, args.num_envs, residual_policy.obs_dim))
    actions = torch.zeros((args.num_steps, args.num_envs) + env.action_space.shape)
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))

    # # Get the dataloader for the demo data for the behavior cloning
    # demo_data_loader = get_demo_data_loader(
    #     control_mode=action_type,
    #     n_batches=args.num_minibatches,
    #     task=args.exp_name,
    #     act_rot_repr=act_rot_repr,
    #     normalizer=normalizer,
    #     action_horizon=agent.action_horizon,
    #     num_workers=4 if not args.debug else 0,
    #     add_relative_pose=args.add_relative_pose,
    # )

    # # Print the number of batches in the dataloader
    # print(f"Number of batches in the dataloader: {len(demo_data_loader)}")

    global_step = 0
    start_time = time.time()
    # bp()
    next_done = torch.zeros(args.num_envs)
    next_obs = env.reset()
    base_observation_deque = deque(
        [next_obs for _ in range(bc_actor.obs_horizon)],
        maxlen=bc_actor.obs_horizon,
    )

    # First get the base normalized action
    base_naction = bc_actor.action_normalized(base_observation_deque)

    # Make the observation for the residual policy by concatenating the state and the base action
    next_obs = env.process_obs(next_obs)
    next_residual_obs = torch.cat([next_obs, base_naction], dim=-1)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iteration: {iteration}/{args.num_iterations}")
        print(f"Run name: {run_name}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            if iteration == 1 and args.bc_coef == 1 and False:
                # Skip the first step to avoid the initial randomness if we're only doing BC
                continue

            global_step += args.num_envs
            dones[step] = next_done
            obs[step] = next_residual_obs

            with torch.no_grad():
                residual_naction, logprob, _, value, _ = (
                    residual_policy.get_action_and_value(next_residual_obs)
                )
                values[step] = value.flatten().cpu()

            naction = base_naction + residual_naction / 10.0

            next_obs, reward, next_done, truncated, infos = env.step(naction)

            # Add the observation to the deque
            base_observation_deque.append(next_obs)

            # Get the base normalized action
            base_naction = bc_actor.action_normalized(base_observation_deque)

            # Process the obs for the residual policy
            next_obs = env.process_obs(next_obs)

            next_residual_obs = torch.cat([next_obs, base_naction], dim=-1)

            actions[step] = residual_naction.cpu()
            logprobs[step] = logprob.cpu()

            rewards[step] = reward.view(-1).cpu()
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()}"
                )

        # Calculate the discounted rewards
        # TODO: Change this so that it takes into account cyclic resets and multiple episodes
        # per experience collection iteration
        discounted_rewards = (
            (rewards * args.gamma ** torch.arange(args.num_steps).float().unsqueeze(1))
            .sum(dim=0)
            .mean()
            .item()
        )
        # If any positive reward was received, consider it a success
        reward_mask = (rewards.sum(dim=0) > 0).float()
        success_rate = reward_mask.mean().item()

        running_mean_success_rate = 0.5 * running_mean_success_rate + 0.5 * success_rate

        print(
            f"Mean return: {discounted_rewards:.4f}, SR: {success_rate:.4%}, SR mean: {running_mean_success_rate:.4%}"
        )

        b_obs = obs.reshape((-1, residual_policy.obs_dim))
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)
        b_values = values.reshape(-1)

        rounded_list = lambda x: [round(i, 3) for i in x.tolist()]

        # Print some stats on the actions stored from the rollouts
        print(f"Action mean: {rounded_list(b_actions.mean(dim=0))}")
        print(f"Action std: {rounded_list(b_actions.std(dim=0))}")
        print(f"Action max: {rounded_list(b_actions.max(dim=0).values)}")
        print(f"Action min: {rounded_list(b_actions.min(dim=0).values)}")

        # bootstrap value if not done
        b_advantages, b_returns = calculate_advantage(
            args,
            device,
            residual_policy,
            b_obs.view((args.num_steps, args.num_envs, residual_policy.obs_dim)).to(
                device
            ),
            next_residual_obs.to(device),
            b_rewards.view(args.num_steps, -1).to(device),
            b_dones.view(args.num_steps, -1).to(device),
        )
        b_advantages = b_advantages.reshape(-1).cpu()
        b_returns = b_returns.reshape(-1).cpu()

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in trange(args.update_epochs, desc="Policy update"):
            # if args.bc_coef > 0:
            #     demo_data_iter = iter(demo_data_loader)

            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Get the minibatch and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)

                # Calculate the loss
                _, newlogprob, entropy, newvalue, action_mean = (
                    residual_policy.get_action_and_value(mb_obs, mb_actions)
                )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                policy_loss = 0

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean() * args.ent_coef

                ppo_loss = pg_loss - entropy_loss
                # Add the auxiliary regularization loss
                aux_loss = torch.mean(torch.square(action_mean))

                # Normalize the losses so that each term has the same scale
                if iteration > args.n_iterations_train_only_value:
                    # Calculate the scaling factors based on the magnitudes of the losses
                    ppo_loss_scale = 1.0 / (torch.abs(ppo_loss.detach()) + 1e-8)
                    aux_loss_scale = 1.0 / (torch.abs(aux_loss.detach()) + 1e-8)

                    # Scale the losses using the calculated scaling factors
                    policy_loss += ppo_loss * ppo_loss_scale
                    policy_loss += (
                        args.residual_regularization * aux_loss * aux_loss_scale
                    )

                # Scale the value loss
                v_loss_scale = 1.0 / (torch.abs(v_loss.detach()) + 1e-8)
                scaled_v_loss = args.vf_coef * v_loss * v_loss_scale

                # Total loss
                loss = policy_loss + scaled_v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    residual_policy.parameters(), args.max_grad_norm
                )
                optimizer.step()

            if (
                args.target_kl is not None
                and approx_kl > args.target_kl
                and args.bc_coef < 1.0
            ):
                print(
                    f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {args.target_kl:.4f}"
                )
                break

            # # Recalculate the advantages with the updated policy before the next epoch
            # if args.recalculate_advantages:
            #     b_advantages, b_returns = calculate_advantage(
            #         args,
            #         device,
            #         residual_policy,
            #         b_obs.view(
            #             (args.num_steps, args.num_envs, residual_policy.obs_dim)
            #         ).to(device),
            #         next_obs.to(device),
            #         b_rewards.view(args.num_steps, -1).to(device),
            #         b_dones.view(args.num_steps, -1).to(device),
            #     )
            #     b_advantages = b_advantages.reshape(-1).cpu()
            #     b_returns = b_returns.reshape(-1).cpu()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Calculate return and advantage with the original experience sample for logging
        advantages, returns = calculate_advantage(
            args,
            device,
            residual_policy,
            obs.to(device),
            next_residual_obs.to(device),
            rewards.to(device),
            dones.to(device),
        )
        advantages = advantages.cpu()
        returns = returns.cpu()
        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "losses/value_loss",
            v_loss.item(),
            global_step,
        )
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # if args.bc_coef > 0:
        #     writer.add_scalar("losses/bc_loss", bc_loss.item(), global_step)
        #     writer.add_scalar("losses/bc_v_loss", bc_v_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/bc_coef", args.bc_coef, global_step)
        writer.add_scalar("losses/residual_l2", aux_loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("charts/rewards", rewards.sum().item(), global_step)
        writer.add_scalar("charts/discounted_rewards", discounted_rewards, global_step)
        writer.add_scalar("charts/success_rate", success_rate, global_step)

        # Add the mean and std of the action norms to the tensorboard
        writer.add_scalar("charts/action_norm_mean", action_norms.mean(), global_step)
        writer.add_scalar("charts/action_norm_std", action_norms.std(), global_step)

        # Add histograms
        writer.add_histogram("histograms/values", values, global_step)
        writer.add_histogram("histograms/returns", returns, global_step)
        writer.add_histogram("histograms/advantages", advantages, global_step)
        writer.add_histogram("histograms/logprobs", logprobs, global_step)
        writer.add_histogram("histograms/rewards", rewards, global_step)

        # # Log the current randomness of the environment
        # writer.add_scalar("env/force_magnitude", env.force_magnitude, global_step)
        # writer.add_scalar("env/torque_magnitude", env.torque_magnitude, global_step)

        # Get the norm of the xyz of actions and plot that on a histogram
        writer.add_histogram("histograms/action_norms", action_norms, global_step)

        # # Add histograms for the gradients and the weights
        # for name, param in residual_policy.named_parameters():
        #     writer.add_histogram(f"weights/{name}", param, global_step)
        #     if param.grad is not None:
        #         writer.add_histogram(f"grads/{name}", param.grad, global_step)

        # Checkpoint the model if the success rate improves
        if success_rate > 0.1 and (
            success_rate > best_success_rate
            or discounted_rewards > best_mean_episode_return
        ):
            best_success_rate = max(success_rate, best_success_rate)
            best_mean_episode_return = max(discounted_rewards, best_mean_episode_return)

            model_path = f"{run_directory}/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(residual_policy.state_dict(), model_path)
            print(f"model saved to {model_path}")

    print(f"Training finished in {(time.time() - start_time):.2f}s")

    if args.save_model:
        model_path = f"{run_directory}/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(residual_policy.state_dict(), model_path)
        print(f"model saved to {model_path}")

    writer.close()
