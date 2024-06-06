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
    cache_dir = os.environ.get("WANDB_CACHE_DIR", "./wandb_cache")
    cache_file = os.path.join(cache_dir, f"{run_id.replace('/', '-')}_{wt_type}.pkl")

    if wandb_mode == "offline" and os.path.exists(cache_file):
        # Load the cached data from the file system
        with open(cache_file, "rb") as f:
            cfg, model_path = pickle.load(f)
    else:
        try:
            # Try to fetch the data using the Weights and Biases API
            api = wandb.Api(overrides=dict(entity="robust-assembly"))
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
            os.makedirs(cache_dir, exist_ok=True)
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

    # def make_config_override_actor(
    #     run: Run,
    #     action_horizon: Union[int, None] = None,
    #     inference_steps: Union[int, None] = None,
    # ):
    #     cfg: DictConfig = OmegaConf.create(
    #         {
    #             **run.config,
    #             "project_name": run.project,
    #             "actor": {
    #                 **run.config["actor"],
    #                 "inference_steps": (
    #                     inference_steps if inference_steps is not None else 4
    #                 ),
    #                 "action_horizon": (
    #                     action_horizon
    #                     if action_horizon is not None
    #                     else run.config["actor"]["action_horizon"]
    #                 ),
    #             },
    #         },
    #     )
    #     return cfg

    # if actor_name == "residual_diffusion":
    #     bp()
    #     # if residual, load the config from the base policy and merge
    #     res_cfg: DictConfig = OmegaConf.create(run.config)

    #     api = wandb.Api(overrides=dict(entity="ankile"))
    #     base_run_id = run.config["base_bc_poliy"]
    #     base_run: Run = api.run(base_run_id)
    #     cfg = make_config_override_actor(
    #         base_run, action_horizon=action_horizon, inference_steps=inference_steps
    #     )

    #     # merge
    #     cfg.actor.update({"residual_policy": res_cfg.residual_policy})

    # else:
    #     # if base BC, just directly load the config
    #     cfg = make_config_override_actor(run, action_horizon=action_horizon)

    return run.config


def load_model_weights(
    run: Run, actor: Actor, wt_type: str = "best", device: str = "cuda"
):

    def get_model_path_from_run(run: Run):
        checkpoint_type = wt_type
        model_file = [
            f
            for f in run.files()
            if f.name.endswith(".pt") and checkpoint_type in f.name
        ]

        if len(model_file) == 0:
            print(f"Could not find model file for run {run.name} wts {wt_type}")
            return None

        model_file = model_file[0]

        print(f"Loading checkpoint: {model_file.name}")
        model_path = model_file.download(
            root=f"./models/{run.name}", exist_ok=True, replace=True
        ).name

        print(f"Model path: {model_path}")
        return model_path

    # if "residual" in run.project:
    #     # if residual, load the config from the base policy and merge
    #     res_model_path = get_model_path_from_run(run)
    #     actor.residual_policy.load_state_dict(
    #         torch.load(res_model_path)["model_state_dict"]
    #     )

    #     api = wandb.Api(overrides=dict(entity="ankile"))
    #     base_run_id = run.config["base_bc_poliy"]
    #     base_run: Run = api.run(base_run_id)
    #     base_model_path = get_model_path_from_run(base_run)

    #     base_state_dict = torch.load(base_model_path)

    #     base_model_state_dict = {
    #         key[len("model.") :]: value
    #         for key, value in base_state_dict.items()
    #         if key.startswith("model.")
    #     }
    #     base_normalizer_state_dict = {
    #         key[len("normalizer.") :]: value
    #         for key, value in base_state_dict.items()
    #         if key.startswith("normalizer.")
    #     }

    #     # Load the normalizer state dict
    #     actor.normalizer.load_state_dict(base_normalizer_state_dict)
    #     actor.model.load_state_dict(base_model_state_dict)
    #     # actor.model.load_state_dict(torch.load(base_model_path))
    # else:

    model_path = get_model_path_from_run(run)

    if model_path is None:
        return None

    state_dict = torch.load(model_path)

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    actor.load_state_dict(state_dict)

    actor.eval()
    actor.to(device)

    return actor
