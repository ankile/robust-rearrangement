from collections import deque
import torch
import torch.nn as nn
from src.data.normalizer import StateActionNormalizer
from src.models.vision import get_encoder
from src.models.unet import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.models.value import DoubleCritic, ValueNetwork
from src.behavior.base import Actor

from ipdb import set_trace as bp  # noqa
from typing import Union


class DiffusionPolicy(Actor):
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
        self.freeze_encoder = freeze_encoder
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
            self.encoder1
            if freeze_encoder
            else get_encoder(encoder_name, freeze=freeze_encoder, device=device)
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

    def __post_init__(self, *args, **kwargs):
        self.print_model_params()

    def print_model_params(self: torch.nn.Module):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:.2e}")

        for name, submodule in self.named_children():
            params = sum(p.numel() for p in submodule.parameters())
            print(f"{name}: {params:.2e} parameters")

    # === Inference ===
    def _normalized_action(self, nobs):
        B = nobs.shape[0]
        # Important! `nobs` needs to be normalized before passing to this function
        # Initialize action from Guassian noise
        naction = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=self.device,
        )

        # init scheduler
        self.inference_noise_scheduler.set_timesteps(self.inference_steps)

        for k in self.inference_noise_scheduler.timesteps:
            # predict noise
            # Print dtypes of all tensors to the model
            noise_pred = self.model(sample=naction, timestep=k, global_cond=nobs)

            # inverse diffusion step (remove noise)
            naction = self.inference_noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        return naction

    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations
        nobs = self._normalized_obs(obs)

        # Predict normalized action
        naction = self._normalized_action(nobs)

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
