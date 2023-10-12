from collections import deque
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import torchvision
from functools import partial
from src.data.dataset import normalize_data, unnormalize_data
from src.models.vision import ResnetEncoder, get_encoder
from src.models.unet import ConditionalUnet1D
from diffusers.optimization import get_scheduler

from src.common.pytorch_util import replace_submodules
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from ipdb import set_trace as bp
import numpy as np
from src.models.module_attr_mixin import ModuleAttrMixin
from typing import Any, Optional, Union
from src.common.pytorch_util import dict_apply

from src.gym import get_env


import lightning.pytorch as pl


class LitImageActor(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str,
        config,
        stats,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        self.obs_horizon = config.obs_horizon
        self.inference_steps = config.inference_steps
        # This is the number of environments only used for inference, not training
        # Maybe it makes sense to do this another way
        self.B = config.num_envs

        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule=config.beta_schedule,
            # clip output to [-1,1] to improve stability
            clip_sample=config.clip_sample,
            # our network predicts noise (instead of denoised action)
            prediction_type=config.prediction_type,
        )

        self.inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )

        # Convert the stats to tensors on the device
        self.stats = self._to_param_dict(stats, requires_grad=False)

        self.encoder1 = get_encoder(encoder_name, freeze=False)
        self.encoder2 = get_encoder(encoder_name, freeze=False)

        self.encoding_dim = self.encoder1.encoding_dim + self.encoder2.encoding_dim
        self.obs_dim = config.robot_state_dim + self.encoding_dim

        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=self.obs_dim * config.obs_horizon,
            down_dims=config.down_dims,
        )

    def _to_param_dict(self, d, requires_grad=False):
        root_module = nn.ParameterDict()

        for key, value in d.items():
            if isinstance(value, dict):
                root_module[key] = self._to_param_dict(value)
            else:
                root_module[key] = torch.from_numpy(value)

        for param in root_module.parameters():
            param.requires_grad = requires_grad

        return root_module

    def _normalized_obs(self, obs: deque):
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)
        nrobot_state = normalize_data(robot_state, stats=self.stats["robot_state"])

        # Convert images from obs_horizon x (n_envs, 224, 224, 3) -> (n_envs, obs_horizon, 224, 224, 3)
        # Also flatten the two first dimensions to get (n_envs * obs_horizon, 224, 224, 3) for the encoder
        img1 = torch.cat([o["color_image1"].unsqueeze(1) for o in obs], dim=1).reshape(
            self.B * self.obs_horizon, 224, 224, 3
        )
        img2 = torch.cat([o["color_image2"].unsqueeze(1) for o in obs], dim=1).reshape(
            self.B * self.obs_horizon, 224, 224, 3
        )

        # Encode images
        # TODO: Do we need to reshape the images back to (n_envs, obs_horizon, -1) after encoding?
        # Probably not because we're flattening the robot_state above also
        features1 = self.encoder1(img1).reshape(self.B, self.obs_horizon, -1)
        features2 = self.encoder2(img2).reshape(self.B, self.obs_horizon, -1)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, features1, features2], dim=-1).flatten(
            start_dim=1
        )

        return nobs

    # === Inference ===
    @torch.no_grad()
    def action(self, obs: deque):
        obs_cond = self._normalized_obs(obs)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (self.B, self.pred_horizon, self.action_dim),
            device=self.device,
        )
        naction = noisy_action

        # init scheduler
        self.inference_noise_scheduler.set_timesteps(self.inference_steps)

        for k in self.inference_noise_scheduler.timesteps:
            # predict noise
            # Print dtypes of all tensors to the model
            noise_pred = self.model(sample=naction, timestep=k, global_cond=obs_cond)

            # inverse diffusion step (remove noise)
            naction = self.inference_noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        # unnormalize action
        # (B, pred_horizon, action_dim)
        # naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])

        return action_pred

    # === Training ===
    def training_step(self, batch, batch_idx):
        # Move the batch to the device
        # TODO: Consider, do we even want to evaluate training on precomputed image features
        # or should we use the results from Diffusion Policy and always do end-to-end training?
        # If we want to use precomputed features, we need to implement an actor that accomodates that
        # for training but performs calls the encoder for inference
        nrobot_state = normalize_data(
            batch["robot_state"], stats=self.stats["robot_state"]
        )
        B = nrobot_state.shape[0]

        # Convert images from obs_horizon x (n_envs, 224, 224, 3) -> (n_envs, obs_horizon, 224, 224, 3)
        # so that it's compatible with the encoder
        image1 = batch["color_image1"].reshape(B * self.obs_horizon, 224, 224, 3)
        image2 = batch["color_image2"].reshape(B * self.obs_horizon, 224, 224, 3)

        # Encode images and reshape back to (B, obs_horizon, -1)
        image1 = self.encoder1(image1).reshape(B, self.obs_horizon, -1)
        image2 = self.encoder2(image2).reshape(B, self.obs_horizon, -1)

        # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
        nobs = torch.cat([nrobot_state, image1, image2], dim=-1)

        # observation as FiLM conditioning
        # (B, obs_horizon, obs_dim)
        obs_cond = nobs[:, : self.obs_horizon, :]
        # (B, obs_horizon * obs_dim)
        obs_cond = obs_cond.flatten(start_dim=1)

        # sample noise to add to actions
        naction = normalize_data(batch["action"], stats=self.stats["action"])
        noise = torch.randn(naction.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action = self.train_noise_scheduler.add_noise(naction, noise, timesteps)

        # forward pass
        noise_pred = self.model(noisy_action, timesteps, global_cond=obs_cond.float())
        loss = nn.functional.mse_loss(noise_pred, noise)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config.actor_lr,
            weight_decay=self.config.weight_decay,
        )

        print("[Warning!] Learning rate scheduler not implemented yet")

        # scheduler = get_scheduler(
        #     name=self.config.lr_scheduler_type,
        #     optimizer=optimizer,
        #     num_warmup_steps=self.config.lr_scheduler_warmup_steps,
        #     num_training_steps=config.num_epochs * n_batches,
        # )

        return optimizer


from pytorch_lightning import Callback


class RolloutCallback(Callback):
    def __init__(self, rollout_function):
        self.rollout_function = rollout_function

    def on_validation_epoch_start(self, trainer, pl_module):
        self.rollout_function(pl_module)
