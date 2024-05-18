from omegaconf import DictConfig, OmegaConf
from src.behavior import get_actor
from src.behavior.base import Actor
import torch
import wandb


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
