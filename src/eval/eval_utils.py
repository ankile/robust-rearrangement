from omegaconf import DictConfig, OmegaConf
from src.behavior import get_actor
from src.behavior.base import Actor
import torch
import wandb

from typing import Union
from wandb import Api
from wandb.sdk.wandb_run import Run


def load_bc_actor(run_id: str, wt_type="best_success_rate", device="cuda"):
    api = wandb.Api(overrides=dict(entity="ankile"))
    run = api.run(run_id)

    cfg: DictConfig = OmegaConf.create(run.config)
    if "flatten_obs" not in cfg.actor:
        cfg.actor.flatten_obs = True
    if "predict_past_actions" not in cfg.actor:
        cfg.actor.predict_past_actions = False

    bc_actor: Actor = get_actor(cfg, device=device)

    model_path = (
        [f for f in run.files() if f.name.endswith(".pt") and wt_type in f.name][0]
        .download(exist_ok=True)
        .name
    )

    print(model_path)

    bc_actor.load_state_dict(torch.load(model_path))
    bc_actor.eval()
    bc_actor.to(device)

    return bc_actor


def load_eval_config(
    run: Run,
    actor_name: str,
    action_horizon: Union[int, None] = None,
    inference_steps: Union[int, None] = None,
):

    def make_config_override_actor(
        run: Run,
        action_horizon: Union[int, None] = None,
        inference_steps: Union[int, None] = None,
    ):
        cfg: DictConfig = OmegaConf.create(
            {
                **run.config,
                "project_name": run.project,
                "actor": {
                    **run.config["actor"],
                    "inference_steps": (
                        inference_steps if inference_steps is not None else 4
                    ),
                    "action_horizon": (
                        action_horizon
                        if action_horizon is not None
                        else run.config["actor"]["action_horizon"]
                    ),
                },
            },
        )
        return cfg

    if "residual" in run.project:
        # if residual, load the config from the base policy and merge
        res_cfg: DictConfig = OmegaConf.create(run.config)

        api = wandb.Api(overrides=dict(entity="ankile"))
        base_run_id = run.config["base_bc_poliy"]
        base_run: Run = api.run(base_run_id)
        cfg = make_config_override_actor(
            base_run, action_horizon=action_horizon, inference_steps=inference_steps
        )

        # merge
        cfg.actor.update({"residual_policy": res_cfg.residual_policy})

    else:
        # if base BC, just directly load the config
        cfg = make_config_override_actor(run, action_horizon=action_horizon)

    cfg.actor.name = actor_name

    return cfg


def load_model_weights(
    run: Run, actor: Actor, wt_type: str = "best", device: str = "cuda"
):

    def get_model_path_from_run(run: Run):
        checkpoint_type = wt_type
        model_file = [
            f
            for f in run.files()
            if f.name.endswith(".pt") and checkpoint_type in f.name
        ][0]
        print(f"Loading checkpoint: {model_file.name}")
        model_path = model_file.download(
            root=f"./models/{run.name}", exist_ok=True, replace=True
        ).name

        print(f"Model path: {model_path}")
        return model_path

    if "residual" in run.project:
        # if residual, load the config from the base policy and merge
        res_model_path = get_model_path_from_run(run)
        actor.residual_policy.load_state_dict(
            torch.load(res_model_path)["model_state_dict"]
        )

        api = wandb.Api(overrides=dict(entity="ankile"))
        base_run_id = run.config["base_bc_poliy"]
        base_run: Run = api.run(base_run_id)
        base_model_path = get_model_path_from_run(base_run)

        base_state_dict = torch.load(base_model_path)

        base_model_state_dict = {
            key[len("model.") :]: value
            for key, value in base_state_dict.items()
            if key.startswith("model.")
        }
        base_normalizer_state_dict = {
            key[len("normalizer.") :]: value
            for key, value in base_state_dict.items()
            if key.startswith("normalizer.")
        }

        # Load the normalizer state dict
        actor.normalizer.load_state_dict(base_normalizer_state_dict)
        actor.model.load_state_dict(base_model_state_dict)
        # actor.model.load_state_dict(torch.load(base_model_path))
    else:
        model_path = get_model_path_from_run(run)
        actor.load_state_dict(torch.load(model_path))

    actor.eval()
    actor.to(device)

    return actor
