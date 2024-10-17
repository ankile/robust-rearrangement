from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.behavior import get_actor
from src.behavior.base import Actor
import torch
import wandb

from typing import Union
from wandb.sdk.wandb_run import Run

from ipdb import set_trace as bp  # noqa


import os
import pickle
from omegaconf import OmegaConf, DictConfig


def load_bc_actor(run_id: str, wt_type="best_success_rate", device="cuda"):
    cfg, model_path = get_model_from_api_or_cached(run_id, wt_type)

    if "flatten_obs" not in cfg.actor:
        cfg.actor.flatten_obs = True
    if "predict_past_actions" not in cfg.actor:
        cfg.actor.predict_past_actions = False

    bc_actor: Actor = get_actor(cfg, device=device)
    bc_actor.load_state_dict(torch.load(model_path))
    bc_actor.eval()
    bc_actor.to(device)

    return bc_actor


def get_model_from_api_or_cached(run_id: str, wt_type: str, wandb_mode="online"):
    cache_dir = Path(os.environ.get("WANDB_CACHE_DIR", "./wandb_cache")) / "model_wts"
    cache_file = cache_dir / f"{run_id.replace('/', '-')}_{wt_type}.pkl"

    if wandb_mode == "offline" and cache_file.exists():
        # Load the cached data from the file system
        with open(cache_file, "rb") as f:
            cfg, model_path = pickle.load(f)
    else:
        try:
            # Try to fetch the data using the Weights and Biases API
            api = wandb.Api()
            run = api.run(run_id)

            cfg: DictConfig = OmegaConf.create(run.config)

            if wt_type == "latest":
                model_path = (
                    (
                        sorted(
                            [f for f in run.files() if f.name.endswith(".pt")],
                            key=lambda x: x.updated_at,
                        )[-1]
                    )
                    .download(exist_ok=True, replace=True, root=cache_dir)
                    .name
                )
            elif wt_type is None:
                model_path = None
            else:
                model_path = (
                    [
                        f
                        for f in run.files()
                        if f.name.endswith(".pt") and wt_type in f.name
                    ][0]
                    .download(exist_ok=True, replace=True, root=cache_dir)
                    .name
                )

            # Cache the data on the file system for future use
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump((cfg, model_path), f)

        except Exception as e:
            # If the API call fails, try to find the files on the file system
            print(f"API call failed: {str(e)}. Searching for files on the file system.")
            cfg_path = os.path.join(run_id, "config.yaml")
            model_path = os.path.join(run_id, f"{wt_type}.pt")

            if os.path.exists(cfg_path) and os.path.exists(model_path):
                cfg = OmegaConf.load(cfg_path)
            else:
                raise FileNotFoundError(
                    f"Could not find the required files for run {run_id}"
                )

    return cfg, model_path


def load_eval_config(
    run: Run,
    actor_name: str,
    action_horizon: Union[int, None] = None,
    inference_steps: Union[int, None] = None,
):

    return run.config


def load_model_weights(
    run: Run, actor: Actor, wt_type: str = "best", device: str = "cuda"
):

    _, model_path = get_model_from_api_or_cached(f"{run.project}/{run.id}", wt_type)

    if model_path is None:
        return None

    state_dict = torch.load(model_path)

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    actor.load_state_dict(state_dict)

    actor.eval()
    actor.to(device)

    return actor
