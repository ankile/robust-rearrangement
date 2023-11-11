from collections import deque
import torch
import torch.nn as nn
from torchvision import transforms
from functools import partial
from src.data.normalizer import StateActionNormalizer
from src.models.vision import ResnetEncoder, get_encoder
from src.models.unet import ConditionalUnet1D
from src.common.pytorch_util import replace_submodules
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchvision.transforms.functional as F
from src.models.value import DoubleCritic, ValueNetwork

from ipdb import set_trace as bp
import numpy as np
from src.models.module_attr_mixin import ModuleAttrMixin
from typing import Union
from src.common.pytorch_util import dict_apply


class DoubleImageActor(torch.nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: StateActionNormalizer,
        config,
    ) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        self.action_horizon = config.action_horizon
        self.obs_horizon = config.obs_horizon
        self.inference_steps = config.inference_steps
        self.observation_type = config.observation_type
        self.noise_augment = config.noise_augment
        # This is the number of environments only used for inference, not training
        # Maybe it makes sense to do this another way
        self.B = config.num_envs
        self.device = device

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
        self.normalizer = normalizer.to(device)

        self.encoder1 = get_encoder(encoder_name, freeze=freeze_encoder, device=device)
        self.encoder2 = (
            get_encoder(encoder_name, freeze=freeze_encoder, device=device)
            if not freeze_encoder
            else self.encoder1
        )

        self.encoding_dim = self.encoder1.encoding_dim + self.encoder2.encoding_dim
        self.timestep_obs_dim = config.robot_state_dim + self.encoding_dim
        self.obs_dim = self.timestep_obs_dim * self.obs_horizon

        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=self.obs_dim,
            down_dims=config.down_dims,
        ).to(device)

        self.dropout = (
            nn.Dropout(config.feature_dropout) if config.feature_dropout else None
        )

        self.print_model_params()

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:.2e}")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params:.2e} parameters")

    def _normalized_obs(self, obs: deque):
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        # Get size of the image
        img_size = obs[0]["color_image1"].shape[-3:]

        # Images come in as obs_horizon x (n_envs, 224, 224, 3) concatenate to (n_envs * obs_horizon, 224, 224, 3)
        img1 = torch.cat([o["color_image1"].unsqueeze(1) for o in obs], dim=1).reshape(
            self.B * self.obs_horizon, *img_size
        )
        img2 = torch.cat([o["color_image2"].unsqueeze(1) for o in obs], dim=1).reshape(
            self.B * self.obs_horizon, *img_size
        )

        # Encode the images and reshape back to (B, obs_horizon, -1)
        features1 = self.encoder1(img1).reshape(self.B, self.obs_horizon, -1)
        features2 = self.encoder2(img2).reshape(self.B, self.obs_horizon, -1)

        if "feature1" in self.normalizer.stats:
            features1 = self.normalizer(features1, "feature1", forward=True)
            features2 = self.normalizer(features2, "feature2", forward=True)

        # Reshape concatenate the features
        nobs = torch.cat([nrobot_state, features1, features2], dim=-1)
        nobs = nobs.flatten(start_dim=1)

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
        action_pred = self.normalizer(naction, "action", forward=False)

        return action_pred

    # === Training ===
    def compute_loss(self, batch):
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch)

        # Apply Dropout to the observation conditioning if specified
        if self.dropout:
            obs_cond = self.dropout(obs_cond)

        # Action already normalized in the dataset
        # naction = normalize_data(batch["action"], stats=self.stats["action"])
        naction = batch["action"]
        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (obs_cond.shape[0],),
            device=self.device,
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action = self.train_noise_scheduler.add_noise(naction, noise, timesteps)

        # forward pass
        noise_pred = self.model(noisy_action, timesteps, global_cond=obs_cond.float())
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    def _training_obs(self, batch):
        nrobot_state = batch["robot_state"]
        B = nrobot_state.shape[0]

        if self.observation_type == "image":
            # Convert images from obs_horizon x (n_envs, 224, 224, 3) -> (n_envs, obs_horizon, 224, 224, 3)
            # so that it's compatible with the encoder
            image1 = batch["color_image1"].reshape(B * self.obs_horizon, 224, 224, 3)
            image2 = batch["color_image2"].reshape(B * self.obs_horizon, 224, 224, 3)

            # Encode images and reshape back to (B, obs_horizon, -1)
            image1 = self.encoder1(image1).reshape(B, self.obs_horizon, -1)
            image2 = self.encoder2(image2).reshape(B, self.obs_horizon, -1)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, image1, image2], dim=-1)

        elif self.observation_type == "feature":
            # All observations already normalized in the dataset
            feature1 = batch["feature1"]
            feature2 = batch["feature2"]

            if self.noise_augment:
                feature1 += torch.normal(
                    mean=0.0, std=self.noise_augment, size=feature1.size()
                ).to(feature1.device)
                feature2 += torch.normal(
                    mean=0.0, std=self.noise_augment, size=feature2.size()
                ).to(feature2.device)

            # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
            nobs = torch.cat([nrobot_state, feature1, feature2], dim=-1)
            nobs = nobs.flatten(start_dim=1)
        return nobs


class ImplicitQActor(DoubleImageActor):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: StateActionNormalizer,
        config,
    ) -> None:
        super().__init__(device, encoder_name, freeze_encoder, normalizer, config)

        # Add hyperparameters specific to IDQL
        self.expectile = config.expectile
        self.tau = config.q_target_update_step
        self.discount = config.discount
        # self.temperature = None

        # Add networks for the Q function
        self.q_network = DoubleCritic(
            state_dim=self.obs_dim,
            action_dim=self.action_dim * self.action_horizon,
            hidden_dims=config.critic_hidden_dims,
            dropout=config.critic_dropout,
        ).to(device)

        self.q_target_network = DoubleCritic(
            state_dim=self.obs_dim,
            action_dim=self.action_dim * self.action_horizon,
            hidden_dims=config.critic_hidden_dims,
            dropout=config.critic_dropout,
        ).to(device)

        # Turn off gradients for the target network
        for param in self.q_target_network.parameters():
            param.requires_grad = False

        # Add networks for the value function
        self.value_network = ValueNetwork(
            input_dim=self.obs_dim,
            hidden_dims=config.critic_hidden_dims,
            dropout=config.critic_dropout,
        ).to(device)

    def _flat_action(self, action):
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        naction = action[:, start:end, :].flatten(start_dim=1)
        return naction

    def _value_loss(self, batch):
        def loss(diff, expectile=0.8):
            weight = torch.where(
                diff > 0,
                torch.full_like(diff, expectile),
                torch.full_like(diff, 1 - expectile),
            )
            return weight * (diff**2)

        # Compute the value loss
        nobs = self._training_obs(batch["curr_obs"])
        naction = self._flat_action(batch["action"])

        # Compute the Q values
        with torch.no_grad():
            q1, q2 = self.q_target_network(nobs, naction)
            q = torch.min(q1, q2)

        v = self.value_network(nobs)

        # Compute the value loss
        value_loss = loss(q - v, expectile=self.expectile).mean()

        return value_loss

    def _q_loss(self, batch):
        curr_obs = self._training_obs(batch["curr_obs"])
        next_obs = self._training_obs(batch["next_obs"])
        naction = self._flat_action(batch["action"])

        with torch.no_grad():
            next_v = self.value_network(next_obs).squeeze(-1)

        target_q = batch["reward"] + self.discount * next_v

        q1, q2 = self.q_network(curr_obs, naction)

        q1_loss = nn.functional.mse_loss(q1.squeeze(-1), target_q)
        q2_loss = nn.functional.mse_loss(q2.squeeze(-1), target_q)

        return (q1_loss + q2_loss) / 2

    def compute_loss(self, batch):
        bc_loss = super().compute_loss({**batch["curr_obs"], "action": batch["action"]})
        q_loss = self._q_loss(batch)
        value_loss = self._value_loss(batch)

        return bc_loss, q_loss, value_loss

    def polyak_update_target(self, tau):
        with torch.no_grad():
            for param, target_param in zip(
                self.q_network.parameters(), self.q_target_network.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
