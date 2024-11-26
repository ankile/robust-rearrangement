from collections import deque
from src.common.geometry import proprioceptive_quat_to_6d_rotation
from omegaconf import DictConfig
from src.models.vision import DualInputAttentionPool2d
import torch
import torch.nn as nn

from src.dataset.normalizer import LinearNormalizer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.behavior.base import Actor
from src.models import get_diffusion_backbone

from ipdb import set_trace as bp  # noqa
from typing import Tuple, Union


class DiffusionPolicy(Actor):

    def __init__(
        self,
        device: Union[str, torch.device],
        cfg: DictConfig,
    ) -> None:
        super().__init__(device, cfg)
        actor_cfg = cfg.actor

        # Diffusion-specific parameters
        self.inference_steps = actor_cfg.inference_steps
        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=actor_cfg.num_diffusion_iters,
            beta_schedule=actor_cfg.beta_schedule,
            clip_sample=actor_cfg.clip_sample,
            prediction_type=actor_cfg.prediction_type,
        )

        self.inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=actor_cfg.num_diffusion_iters,
            beta_schedule=actor_cfg.beta_schedule,
            clip_sample=actor_cfg.clip_sample,
            prediction_type=actor_cfg.prediction_type,
        )

        self.model = get_diffusion_backbone(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            actor_config=actor_cfg,
        ).to(device)

        self.warmstart_timestep = 50
        self.prev_naction = None
        self.eta = 0.0

    # === Inference ===
    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """
        Perform diffusion to generate actions given the observation.
        """
        B = nobs.shape[0]

        if not self.flatten_obs and len(nobs.shape) == 2:
            # If the observation is not flattened, we need to reshape it to (B, obs_horizon, obs_dim)
            nobs = nobs.reshape(B, self.obs_horizon, self.obs_dim)

        # Now we know what batch size we have, so set the previous action to zeros of the correct size
        if self.prev_naction is None or self.prev_naction.shape[0] != B:
            self.prev_naction = torch.zeros(
                (B, self.pred_horizon, self.action_dim), device=self.device
            )

        # Important! `nobs` needs to be normalized and flattened before passing to this function
        # Sample Gaussian noise to use to corrupt the actions
        noise = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=self.device,
        )

        # init scheduler
        self.inference_noise_scheduler.set_timesteps(self.inference_steps)

        # Instead of sampling noise, we'll start with the previous action and add noise
        naction = self.prev_naction

        naction = self.inference_noise_scheduler.add_noise(
            naction,
            noise,
            torch.full((B,), self.warmstart_timestep, device=self.device).long(),
        )

        for k in self.inference_noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.model(sample=naction, timestep=k, global_cond=nobs)

            # inverse diffusion step (remove noise)
            naction = self.inference_noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
                eta=self.eta,
            ).prev_sample

        # Store the remaining actions in the previous action to warm start the next horizon
        self.prev_naction[:, : self.pred_horizon - self.action_horizon, :] = naction[
            :, self.action_horizon :, :
        ]

        return naction

    # === Training ===
    def compute_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=self.flatten_obs)

        # Action already normalized in the dataset
        # These actions are the exact ones we should predict, i.e., the
        # handling of predicting past actions or not is also handled in the dataset class
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
        noise_pred: torch.Tensor = self.model(
            noisy_action, timesteps, global_cond=obs_cond.float()
        )

        # Without reuction so is of shape (B, H, A)
        loss: torch.Tensor = self.loss_fn(noise_pred, noise)

        # Take the mean over the last two dimensions to get the loss for each example in the batch
        # (B, H, A) -> (B, 1)
        loss = loss.mean(dim=[1, 2]).unsqueeze(1)

        if self.rescale_loss_for_domain:
            # Calculate class weights
            class_sizes = torch.bincount(batch["domain"].squeeze())
            class_weights = torch.pow(class_sizes.float(), -1.0 / 2)
            class_weights = class_weights / class_weights.sum()

            # Apply class weights to the loss
            class_weights = class_weights[batch["domain"]]
            loss *= class_weights

        loss = loss.mean()
        losses = {"bc_loss": loss.item()}

        # Add the VIB loss
        if self.camera_2_vib is not None:
            mu, log_var = batch["mu"], batch["log_var"]
            vib_loss = self.camera_2_vib.kl_divergence(mu, log_var)

            # Clip the VIB loss to prevent it from dominating the total loss
            losses["vib_loss"] = vib_loss.item()
            vib_loss = torch.clamp(vib_loss, max=1)

            # Scale the VIB loss by the beta and add it to the total loss
            loss += self.vib_front_feature_beta * vib_loss

        # Add the confusion loss
        if self.confusion_loss_beta > 0:
            confusion_loss = batch["confusion_loss"]
            losses["confusion_loss"] = confusion_loss.item()

            loss += self.confusion_loss_beta * confusion_loss

        return loss, losses


class AttentionPoolDiffusionPolicy(DiffusionPolicy):
    def __init__(self, device: Union[str, torch.device], cfg: DictConfig) -> None:
        super().__init__(device, cfg)

        # Attention Pooling
        self.attention_pool = DualInputAttentionPool2d(
            spatial_dim=7,
            embed_dim=512,
            num_heads=8,
            output_dim=256,
        ).to(device)

        # Hook into the encoder to replace the avgpool with identity
        self.encoder1.model.convnet.avgpool = nn.Identity()
        self.encoder2.model.convnet.avgpool = nn.Identity()

    # === Training Observations ===
    def _training_obs(self, batch, flatten: bool = True):

        assert self.observation_type == "image"

        # The robot state is already normalized in the dataset
        nrobot_state = batch["robot_state"]
        B = nrobot_state.shape[0]

        image1: torch.Tensor = batch["color_image1"]
        image2: torch.Tensor = batch["color_image2"]

        # Images now have the channels first
        assert image1.shape[-3:] == (3, 240, 320)

        # Reshape the images to (B * obs_horizon, C, H, W) for the encoder
        image1 = image1.reshape(B * self.obs_horizon, *image1.shape[-3:])
        image2 = image2.reshape(B * self.obs_horizon, *image2.shape[-3:])

        # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
        # Since we're in training mode, the transform also performs augmentation
        image1: torch.Tensor = self.camera1_transform(image1)
        image2: torch.Tensor = self.camera2_transform(image2)

        # Encode images and reshape back to (B, obs_horizon, encoding_dim, 7, 7)
        featuremap1 = (
            self.encoder1(image1)
            .reshape(B * self.obs_horizon, 7, 7, 512)
            .permute(0, 3, 1, 2)
        )
        featuremap2 = (
            self.encoder2(image2)
            .reshape(B * self.obs_horizon, 7, 7, 512)
            .permute(0, 3, 1, 2)
        )

        # Apply the attention pooling to the feature maps
        image_features = self.attention_pool(featuremap1, featuremap2).reshape(
            B, self.obs_horizon, 256
        )

        # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
        nobs = torch.cat([nrobot_state, image_features], dim=-1)

        if flatten:
            # (B, obs_horizon, obs_dim) --> (B, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs

    # === Inference Observations ===
    def _normalized_obs(self, obs: deque, flatten: bool = True):
        """
        Normalize the observations

        Takes in a deque of observations and normalizes them
        And concatenates them into a single tensor of shape (n_envs, obs_horizon * obs_dim)
        """
        # Convert robot_state from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        robot_state = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)

        # Convert the robot_state to use rot_6d instead of quaternion
        robot_state = proprioceptive_quat_to_6d_rotation(robot_state)

        # Normalize the robot_state
        nrobot_state = self.normalizer(robot_state, "robot_state", forward=True)

        B = nrobot_state.shape[0]

        assert self.observation_type == "image"

        # Get size of the image
        img_size = obs[0]["color_image1"].shape[-3:]

        # Images come in as obs_horizon x (n_envs, 224, 224, 3) concatenate to (n_envs * obs_horizon, 224, 224, 3)
        image1 = torch.cat(
            [o["color_image1"].unsqueeze(1) for o in obs], dim=1
        ).reshape(B * self.obs_horizon, *img_size)
        image2 = torch.cat(
            [o["color_image2"].unsqueeze(1) for o in obs], dim=1
        ).reshape(B * self.obs_horizon, *img_size)

        # Move the channel to the front (B * obs_horizon, H, W, C) -> (B * obs_horizon, C, H, W)
        image1 = image1.permute(0, 3, 1, 2)
        image2 = image2.permute(0, 3, 1, 2)

        # Apply the transforms to resize the images to 224x224, (B * obs_horizon, C, 224, 224)
        image1: torch.Tensor = self.camera1_transform(image1)
        image2: torch.Tensor = self.camera2_transform(image2)

        # Encode images and reshape back to (B, obs_horizon, encoding_dim, 7, 7)
        featuremap1 = (
            self.encoder1(image1)
            .reshape(B * self.obs_horizon, 7, 7, 512)
            .permute(0, 3, 1, 2)
        )
        featuremap2 = (
            self.encoder2(image2)
            .reshape(B * self.obs_horizon, 7, 7, 512)
            .permute(0, 3, 1, 2)
        )

        # Apply the attention pooling to the feature maps
        image_features = self.attention_pool(featuremap1, featuremap2).reshape(
            B, self.obs_horizon, 256
        )

        # Combine the robot_state and image features, (B, obs_horizon, obs_dim)
        nobs = torch.cat([nrobot_state, image_features], dim=-1)

        if flatten:
            # (B, obs_horizon, obs_dim) --> (B, obs_horizon * obs_dim)
            nobs = nobs.flatten(start_dim=1)

        return nobs
