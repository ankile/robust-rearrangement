from typing import List
import furniture_bench
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from src.behavior.base import Actor  # noqa
import torch
from omegaconf import OmegaConf, DictConfig
from src.eval.rollout import calculate_success_rate
from src.behavior import get_actor
from src.common.tasks import task2idx
from src.gym import get_env
from src.dataset import get_normalizer

from ipdb import set_trace as bp  # noqa
import wandb
from wandb import Api
from wandb.sdk.wandb_run import Run


import sys
import torch
from typing import List
from wandb.apis import public

api = public.Api()


def convert_state_dict(state_dict):
    if not any(k.startswith("encoder1.0") for k in state_dict.keys()) and not any(
        k.startswith("encoder1.model.nets.3") for k in state_dict.keys()
    ):
        print("Dict already in the correct format")
        return

    for k in list(state_dict.keys()):
        if k.startswith("encoder1.0"):
            new_k = k.replace("encoder1.0", "encoder1")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.0"):
            new_k = k.replace("encoder2.0", "encoder2")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder1.1"):
            new_k = k.replace("encoder1.1", "encoder1_proj")
            state_dict[new_k] = state_dict.pop(k)
        elif k.startswith("encoder2.1"):
            new_k = k.replace("encoder2.1", "encoder2_proj")
            state_dict[new_k] = state_dict.pop(k)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <param1> <param2> <param3> <param4>")
        sys.exit(1)

    kwargs = {}
    kwargs["pos_scalar"] = float(sys.argv[1])
    kwargs["rot_scalar"] = float(sys.argv[2])
    kwargs["stiffness"] = float(sys.argv[3])
    kwargs["damping"] = float(sys.argv[4])

    # Hardcode the necessary variables here
    gpu = 0
    n_envs = 1024
    n_rollouts = n_envs
    randomness = "low"
    furniture = "one_leg"
    n_parts_assemble = None
    action_type = "pos"
    compress_pickles = False
    max_rollout_steps = 650
    no_april_tags = True

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    rollout_max_steps = max_rollout_steps

    env: FurnitureSimEnv = get_env(
        gpu_id=gpu,
        furniture=furniture,
        num_envs=n_envs,
        randomness=randomness,
        observation_space="state",
        max_env_steps=5_000,
        resize_img=False,
        act_rot_repr="rot_6d",
        ctrl_mode="diffik",
        action_type=action_type,
        april_tags=not no_april_tags,
        verbose=False,
        headless=False,
        **kwargs,
    )

    run_id = "ankile/one_leg-mlp-state-1/runs/1pyfen30"
    run = api.run(run_id)

    model_file = [f for f in run.files() if f.name.endswith(".pt")][0]
    model_path = model_file.download(
        root=f"./models/{run.name}", exist_ok=True, replace=True
    ).name

    print(f"Model path: {model_path}")

    config: DictConfig = OmegaConf.create(
        {
            **run.config,
            "project_name": run.project,
            "actor": {**run.config["actor"], "inference_steps": 8},
        },
        flags={"readonly": True},
    )

    normalizer_type = config.get("data", {}).get("normalization", "min_max")
    normalizer = get_normalizer(
        normalizer_type=normalizer_type, control_mode=config.control.control_mode
    )

    actor: Actor = get_actor(cfg=config, normalizer=normalizer, device=device)

    print("NBNB: This is a hack to load the model weights, please fix soon")
    # TODO: Fix this properly, but for now have an ugly escape hatch
    import torch.nn as nn

    actor.normalizer.stats["parts_poses"] = nn.ParameterDict(
        {
            "min": nn.Parameter(torch.zeros(35)),
            "max": nn.Parameter(torch.ones(35)),
        }
    )

    state_dict = torch.load(model_path)

    actor_state_dict = actor.state_dict()

    # Load the model weights
    convert_state_dict(state_dict)

    actor.load_state_dict(state_dict)
    actor.eval()
    actor.cuda()

    actor.set_task(task2idx[furniture])
    rollout_stats = calculate_success_rate(
        actor=actor,
        env=env,
        n_rollouts=n_rollouts,
        rollout_max_steps=rollout_max_steps,
        epoch_idx=0,
        gamma=config.discount,
        rollout_save_dir=None,
        save_failures=False,
        n_parts_assemble=n_parts_assemble,
        compress_pickles=compress_pickles,
        resize_video=True,
    )

    success_rate = rollout_stats.success_rate

    print(
        f"Success rate: {success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})"
    )
    print(float(success_rate * 100))
