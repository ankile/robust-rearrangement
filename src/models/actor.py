from collections import deque
import torch
import torch.nn as nn
import torchvision
from functools import partial
from src.data.dataset import normalize_data, unnormalize_data
from src.models.vision import ResnetEncoder, get_encoder
from src.models.unet import ConditionalUnet1D
from src.common.pytorch_util import replace_submodules
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from ipdb import set_trace as bp
import numpy as np
from src.models.module_attr_mixin import ModuleAttrMixin
from typing import Union
from src.common.pytorch_util import dict_apply


class Actor(torch.nn.Module):
    def __init__(self, noise_net, config, stats) -> None:
        self.noise_net = noise_net
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        self.obs_horizon = config.obs_horizon
        self.inference_steps = config.inference_steps
        self.device = next(noise_net.parameters()).device
        # This is the number of environments only used for inference, not training
        # Maybe it makes sense to do this another way
        self.B = config.num_envs

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )

        # Convert the stats to tensors on the device
        self.stats = {
            "obs": {
                "min": torch.from_numpy(stats["obs"]["min"]).to(self.device),
                "max": torch.from_numpy(stats["obs"]["max"]).to(self.device),
            },
            "action": {
                "min": torch.from_numpy(stats["action"]["min"]).to(self.device),
                "max": torch.from_numpy(stats["action"]["max"]).to(self.device),
            },
        }

    def _normalized_obs(self, obs):
        raise NotImplementedError

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
        self.noise_scheduler.set_timesteps(self.inference_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            # Print dtypes of all tensors to the model
            noise_pred = self.noise_net(
                sample=naction, timestep=k, global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        # unnormalize action
        # (B, pred_horizon, action_dim)
        # naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])

        return action_pred


class StateActor(Actor):
    def _normalized_obs(self, obs):
        agent_pos = torch.from_numpy(
            np.concatenate(
                [o["robot_state"].reshape(self.B, 1, -1) for o in obs],
                axis=1,
            )
        )
        feature1 = np.concatenate(
            [o["image1"].reshape(self.B, 1, -1) for o in obs], axis=1
        )
        feature2 = np.concatenate(
            [o["image2"].reshape(self.B, 1, -1) for o in obs], axis=1
        )
        nobs = torch.from_numpy(
            np.concatenate([agent_pos, feature1, feature2], axis=-1)
        )
        nobs = (
            normalize_data(nobs, stats=self.stats["obs"])
            .flatten(start_dim=1)
            .to(self.device)
        )

        return nobs


class ImageActor(Actor):
    def __init__(self, noise_net, encoder, config, stats) -> None:
        super().__init__(noise_net, config, stats)
        self.encoder = encoder

    def _normalized_obs(self, obs: deque):
        # Convert agent_pos from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        agent_pos = torch.cat([o["agent_pos"].unsqueeze(1) for o in obs], dim=1)

        # Convert images from obs_horizon x (n_envs, 224, 224, 3) -> (n_envs, obs_horizon, 224, 224, 3)
        # Also flatten the two first dimensions to get (n_envs * obs_horizon, 224, 224, 3) for the encoder
        img1 = torch.cat([o["image1"].unsqueeze(1) for o in obs], dim=1).reshape(
            self.B * self.obs_horizon, 224, 224, 3
        )
        img2 = torch.cat([o["image2"].unsqueeze(1) for o in obs], dim=1).reshape(
            self.B * self.obs_horizon, 224, 224, 3
        )

        # Concat images to only do one forward pass through the encoder
        images = torch.cat([img1, img2], dim=0)

        # Encode images
        features = self.encoder(images)
        feature1 = features[: self.B * self.obs_horizon].reshape(
            self.B, self.obs_horizon, -1
        )
        feature2 = features[self.B * self.obs_horizon :].reshape(
            self.B, self.obs_horizon, -1
        )

        nobs = torch.cat([agent_pos, feature1, feature2], dim=-1)
        nobs = normalize_data(nobs, stats=self.stats["obs"]).flatten(start_dim=1)

        return nobs


class DoubleImageActor(torch.nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        config,
        stats,
    ) -> None:
        super().__init__()
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        self.obs_horizon = config.obs_horizon
        self.inference_steps = config.inference_steps
        self.observation_type = config.observation_type
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
        self.stats = dict_apply(stats, lambda x: torch.from_numpy(x).to(device))

        self.encoder1 = get_encoder(encoder_name, freeze=freeze_encoder, device=device)
        self.encoder2 = (
            get_encoder(encoder_name, freeze=freeze_encoder, device=device)
            if not freeze_encoder
            else self.encoder1
        )

        self.encoding_dim = self.encoder1.encoding_dim + self.encoder2.encoding_dim
        self.obs_dim = config.robot_state_dim + self.encoding_dim

        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=self.obs_dim * config.obs_horizon,
            down_dims=config.down_dims,
        ).to(device)

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

        if self.observation_type == "image":
            robot_state = normalize_data(robot_state, stats=self.stats["robot_state"])

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
        nobs = torch.cat([robot_state, features1, features2], dim=-1)

        if self.observation_type == "feature":
            nobs = normalize_data(nobs, stats=self.stats["obs"])

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
        # naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])

        return action_pred

    # === Training ===
    def compute_loss(self, batch):
        # Move the batch to the device
        # TODO: Consider, do we even want to evaluate training on precomputed image features
        # or should we use the results from Diffusion Policy and always do end-to-end training?
        # If we want to use precomputed features, we need to implement an actor that accomodates that
        # for training but performs calls the encoder for inference

        if self.observation_type == "image":
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

        elif self.observation_type == "feature":
            nobs = normalize_data(batch["obs"], stats=self.stats["obs"])
            B = nobs.shape[0]

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

        return loss
