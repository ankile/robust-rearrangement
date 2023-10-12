from pytorch_lightning import Callback
from tqdm import tqdm, trange
import torch
import wandb
import collections
import numpy as np
from src.eval import calculate_success_rate


class RolloutEvaluationCallback(Callback):
    def __init__(self, config, get_env):
        self.config = config
        self.get_env = get_env
        self.best_success_rate = 0.0

    def on_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        
        epoch_idx = trainer.current_epoch
        epoch_loss = # extract or calculate your epoch loss here

        if (
            self.config.rollout_every != -1
            and (epoch_idx + 1) % self.config.rollout_every == 0
            and np.mean(epoch_loss) < self.config.rollout_loss_threshold
        ):
            if hasattr(self, "env"):
                self.env = self.get_env(
                    self.config.gpu_id,
                    obs_type=self.config.observation_type,
                    furniture=self.config.furniture,
                    num_envs=self.config.num_envs,
                    randomness=self.config.randomness,
                )

            success_rate = calculate_success_rate(
                self.env,
                pl_module,  # Assuming 'pl_module' is your actor
                self.config,
                epoch_idx,
            )

            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                save_path = # Your save path logic here
                torch.save(
                    pl_module.state_dict(),
                    save_path,
                )
                wandb.save(save_path)

