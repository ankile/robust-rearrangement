# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os.path
from typing import Dict, Literal
import furniture_bench  # noqa

import random
import time
from dataclasses import dataclass
import math

from src.common.files import get_processed_paths
from src.gym.env_rl_wrapper import FurnitureEnvRLWrapper
from src.gym.furniture_sim_env import (
    FurnitureRLSimEnv,
    FurnitureRLSimEnvPlaceTabletop,
    FurnitureRLSimEnvReacher,
)
from furniture_bench.envs.observation import DEFAULT_STATE_OBS
import numpy as np
from src.common.context import suppress_all_output
from src.common.pytorch_util import dict_to_device
from src.dataset.dataset import FurnitureStateDataset
from src.gym.utils import NormalizeReward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import tyro
from torch.utils.tensorboard import SummaryWriter
from src.behavior.base import Actor
from src.eval.load_model import load_bc_actor

from src.dataset.normalizer import LinearNormalizer
import src.common.geometry as G

from src.behavior.mlp import (
    SmallMLPAgent,
    BigMLPAgent,
    ResidualMLPAgent,
    ResidualMLPAgentSeparate,
)

from ipdb import set_trace as bp

# Set the gym logger to not print to console
import gym

gym.logger.set_level(40)
import wandb


from src.gym import turn_off_april_tags


def get_model_weights(run_id: str):
    api = wandb.Api(overrides=dict(entity="ankile"))
    run = api.run(run_id)
    model_path = (
        [f for f in run.files() if f.name.endswith(".pt")][0]
        .download(exist_ok=True)
        .name
    )
    print(f"Loading checkpoint from {run_id}")
    return torch.load(model_path)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init_zeros(layer):
    torch.nn.init.zeros_(layer.weight)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)
    return layer

class Qfunction(nn.Module):
    def __init__(self, obs_shape, action_shape):
        """
        Args:
            obs_shape: the shape of the observation (i.e., state + base action)
            action_shape: the shape of the action (i.e., residual, same size as base action)
        """
        super().__init__()

        self.action_horizon, self.action_dim = action_shape
        self.obs_dim = np.prod(obs_shape) + np.prod(action_shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init_zeros(nn.Linear(512, 1)),
            nn.Sigmoid()
        )

    def get_value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs)


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
    debug: bool = False
    """if toggled, the debug mode will be enabled"""

    # Algorithm specific arguments
    # env_id: str = "HalfCheetah-v4"
    # """the id of the environment"""
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
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    bc_coef: float = 0.0
    """coefficient of the behavior cloning loss"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    truncation_as_done: bool = True
    """Should we consider truncation due to timelimit as done"""

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

@torch.no_grad()
def calculate_advantage(
    args: Args,
    device: torch.device,
    agent: SmallMLPAgent,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    next_actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor
):
    # bp()
    critic_obs = torch.cat([obs, actions], dim=-1)
    values = agent.get_value(critic_obs).squeeze()
    next_critic_obs = torch.cat([next_obs, next_actions], dim=-1)
    next_value = agent.get_value(next_critic_obs).reshape(1, -1)

    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
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

    run_directory = f"runs/debug-{args.exp_name}-8"
    run_directory += "-delete" if args.debug else ""
    print(f"Run directory: {os.path.abspath(run_directory)}")
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
        elif args.exp_name == "place-tabletop":
            env: FurnitureRLSimEnv = FurnitureRLSimEnvPlaceTabletop(**kwargs)
        elif args.exp_name == "reacher":
            assert args.bc_coef == 0, "Behavior cloning is not supported for Reacher"
            env: FurnitureRLSimEnv = FurnitureRLSimEnvReacher(**kwargs)
        else:
            raise ValueError(f"Unknown experiment name: {args.exp_name}")

    env.max_force_magnitude = 0.1
    env.max_torque_magnitude = 0.005

    env = FurnitureEnvRLWrapper(
        env,
        max_env_steps=args.num_env_steps,
        ee_dof=args.ee_dof,
        chunk_size=args.chunk_size,
        task=args.exp_name,
        add_relative_pose=args.add_relative_pose,
    )

    if args.normalize_reward:
        print("Wrapping the environment with reward normalization")
        env = NormalizeReward(env, device=device)
        normrewenv = env
    else:
        normrewenv = None
        print("Not wrapping the environment with reward normalization")

    if args.agent == "small":
        agent = Qfunction(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    bc_actor: Actor = load_bc_actor("ankile/one_leg-diffusion-state-1/runs/0b7wp7g0")
    bc_actor.eta = 1.0
    env.set_normalizer(bc_actor.normalizer)


    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=1e-5,
    )
    policy_steps = math.ceil(args.num_env_steps / agent.action_horizon)
    args.num_steps = args.data_collection_steps
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(
        f"With chunk size {agent.action_horizon}, we have {policy_steps} policy steps."
    )
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

    obs = torch.zeros((args.num_steps, args.num_envs) + env.observation_space.shape)
    actions = torch.zeros((args.num_steps, args.num_envs) + env.action_space.shape)
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))

    # if args.load_checkpoint:
    #     wts = get_model_weights("one_leg-mlp-state-1/6bh9dn66")
    #
    #     # Filter out keys not starting with "model"
    #     model_wts = {k: v for k, v in wts.items() if k.startswith("model")}
    #     # Change the "model" prefix to "actor_mean"
    #     model_wts = {k.replace("model.", ""): v for k, v in model_wts.items()}
    #
    #     print(agent.actor_mean[0].load_state_dict(model_wts, strict=True))
    #
    #     # Load in the weights for the normalizer
    #     normalizer_wts = {
    #         k.replace("normalizer.", ""): v for k, v in wts.items() if "normalizer" in k
    #     }
    #     print(normalizer.load_state_dict(normalizer_wts))

    agent = agent.to(device)
    bc_actor = bc_actor.to(device)
    bc_actor.eval()

    global_step = 0
    start_time = time.time()
    # bp()
    next_done = torch.zeros(args.num_envs)
    next_obs = env.reset()

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iteration: {iteration}/{args.num_iterations}")
        print(f"Run name: {run_name}")
        agent.eval()

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
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                out = bc_actor._normalized_action(next_obs)
                naction = out[:,:agent.action_horizon,:]
                naction = naction + 0.03 * torch.randn(naction.shape, device=naction.device, )
                critic_obs = torch.cat([next_obs, naction.reshape(args.num_envs,-1)], dim=-1)
                values[step] = agent.get_value(critic_obs).squeeze()

            next_obs, reward, next_done, truncated, infos = env.step(naction)

            actions[step] = naction.cpu()

            rewards[step] = reward.view(-1).cpu()
            if args.truncation_as_done:
                next_done = np.logical_or(next_done.view(-1).cpu(), truncated.view(-1).cpu())
            else:
                next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step * agent.action_horizon) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()}"
                )

        with torch.no_grad():
            out = bc_actor._normalized_action(next_obs)
            next_naction = out[:, :agent.action_horizon, :]
            next_naction = next_naction + 0.01*torch.randn(next_naction.shape, device=next_naction.device,)

        if normrewenv is not None:
            print(f"Current return_rms.var: {normrewenv.return_rms.var.item()}")

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

        b_obs = obs.reshape((-1,) + env.observation_space.shape)
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)
        b_values = values.reshape(-1)

        mask = torch.zeros_like(rewards).to(device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = next_done
            else:
                nextnonterminal = torch.logical_or(dones[t + 1], nextnonterminal)
            mask[t] = nextnonterminal.to(torch.bool)
        b_mask = mask.reshape(-1)

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            args,
            device,
            agent,
            b_obs.view(
                (args.num_steps, args.num_envs) + env.observation_space.shape
            ).to(device),
            b_actions.view((args.num_steps, args.num_envs, int(np.prod(env.action_space.shape)))
            ).to(device),
            next_obs.to(device),
            next_naction.view((args.num_envs, int(np.prod(env.action_space.shape)))
            ).to(device),
            b_rewards.view(args.num_steps, -1).to(device),
            b_dones.view(args.num_steps, -1).to(device),
            next_done.view(args.num_steps).to(device),
        )
        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()

        agent.train()

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in trange(args.update_epochs, desc="Policy update"):

            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                curr_mb_size = mb_inds.shape[0]

                # Get the minibatch and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)
                mb_mask = b_mask[mb_inds].to(device)

                # Calculate the loss
                mb_critic_obs = torch.cat([mb_obs, mb_actions.reshape(curr_mb_size, -1)], dim=-1)
                newvalue = agent.get_value(mb_critic_obs)

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2)

                # denom = torch.sum(mb_mask, -1, keepdim=True)
                # loss = torch.sum(v_loss * mb_mask) / denom
                loss = v_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/y_pred", y_pred.mean().item(), global_step)
        writer.add_scalar("losses/bc_coef", args.bc_coef, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("charts/rewards", rewards.sum().item(), global_step)
        writer.add_scalar("charts/discounted_rewards", discounted_rewards, global_step)
        writer.add_scalar("charts/success_rate", success_rate, global_step)

        # Add histograms
        # returns[~mask.to(torch.bool)] = -1
        writer.add_histogram("histograms/returns", returns, global_step)
        writer.add_histogram("histograms/values", values, global_step)
        writer.add_histogram("histograms/logprobs", logprobs, global_step)
        writer.add_histogram("histograms/rewards", rewards, global_step)

        # Log the current randomness of the environment
        writer.add_scalar("env/force_magnitude", env.force_magnitude, global_step)
        writer.add_scalar("env/torque_magnitude", env.torque_magnitude, global_step)

        # Add histograms for the gradients and the weights
        for name, param in agent.named_parameters():
            writer.add_histogram(f"weights/{name}", param, global_step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, global_step)

        # Checkpoint the model if the success rate improves
        if iteration % 10 == 0:
            model_path = f"{run_directory}/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {os.path.abspath(model_path)}")

    print(f"Training finished in {(time.time() - start_time):.2f}s")

    if args.save_model:
        model_path = f"{run_directory}/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {os.path.abspath(model_path)}")

    writer.close()
