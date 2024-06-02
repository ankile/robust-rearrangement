from collections import deque
from omegaconf import DictConfig
import torch
import torch.nn as nn

from src.dataset.normalizer import LinearNormalizer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.behavior.base import Actor
from src.models import get_diffusion_backbone

from ipdb import set_trace as bp  # noqa
from typing import Union

# our real/debug imports
import pytorch3d.transforms as pt

import wandb


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

    # === Inference ===
    def _normalized_action(self, nobs: torch.Tensor) -> torch.Tensor:
        """
        Perform diffusion to generate actions given the observation.
        """
        B = nobs.shape[0]

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
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        # Store the remaining actions in the previous action to warm start the next horizon
        self.prev_naction[:, : self.pred_horizon - self.action_horizon, :] = naction[
            :, self.action_horizon :, :
        ]

        return naction

    # === Training ===
    def compute_loss(self, batch):
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
        noise_pred = self.model(noisy_action, timesteps, global_cond=obs_cond.float())
        loss = self.loss_fn(noise_pred, noise)

        if self.rescale_loss_for_domain:
            # Calculate class weights
            class_sizes = torch.bincount(batch["domain"])
            class_weights = torch.pow(class_sizes.float(), 1.0 / 3)
            class_weights = class_weights / class_weights.sum()

            # Apply class weights to the loss
            class_weights = class_weights[batch["domain"]]
            scaled_loss = self.loss_fn(noise_pred, noise) * class_weights
            loss = scaled_loss.mean()

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


class MultiTaskDiffusionPolicy(DiffusionPolicy):
    current_task = None

    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: LinearNormalizer,
        cfg,
    ) -> None:
        raise NotImplementedError(
            "Multitask diffusion actor is not supported at the moment."
        )
        super().__init__(
            device=device,
            encoder_name=encoder_name,
            freeze_encoder=freeze_encoder,
            normalizer=normalizer,
            cfg=cfg,
        )

        multitask_cfg = cfg.multitask
        actor_cfg = cfg.actor

        self.task_dim = multitask_cfg.task_dim
        self.task_encoder = nn.Embedding(
            num_embeddings=multitask_cfg.num_tasks,
            embedding_dim=self.task_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
        ).to(device)

        self.obs_dim = self.obs_dim + self.task_dim

        self.model = get_diffusion_backbone(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            actor_config=actor_cfg,
        ).to(device)

    def _training_obs(self, batch, flatten: bool = True):
        # Get the standard observation data
        nobs = super()._training_obs(batch, flatten=flatten)

        # Get the task embedding
        task_idx: torch.Tensor = batch["task_idx"]
        task_embedding: torch.Tensor = self.task_encoder(task_idx)

        if flatten:
            task_embedding = task_embedding.flatten(start_dim=1)

        # Concatenate the task embedding to the observation
        obs_cond = torch.cat((nobs, task_embedding), dim=-1)

        return obs_cond

    def set_task(self, task):
        self.current_task = task

    def _normalized_obs(self, obs: deque, flatten: bool = True):
        assert self.current_task is not None, "Must set current task before calling"

        # Get the standard observation data
        nobs = super()._normalized_obs(obs, flatten=flatten)
        B = nobs.shape[0]

        # Get the task embedding for the current task and repeat it for the batch size and observation horizon
        task_embedding = self.task_encoder(
            torch.tensor(self.current_task).to(self.device)
        ).repeat(B, self.obs_horizon, 1)

        if flatten:
            task_embedding = task_embedding.flatten(start_dim=1)

        # Concatenate the task embedding to the observation
        obs_cond = torch.cat((nobs, task_embedding), dim=-1)

        return obs_cond


class SuccessGuidedDiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: LinearNormalizer,
        cfg,
    ) -> None:
        raise NotImplementedError(
            "Guided diffusion actor is not supported at the moment."
        )
        super().__init__(
            device=device,
            encoder_name=encoder_name,
            freeze_encoder=freeze_encoder,
            normalizer=normalizer,
            cfg=cfg,
        )
        actor_cfg = cfg.actor

        self.guidance_scale = actor_cfg.guidance_scale
        self.prob_blank_cond = actor_cfg.prob_blank_cond
        self.success_cond_emb_dim = actor_cfg.success_cond_emb_dim

        self.success_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.success_cond_emb_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
        ).to(device)

        self.obs_dim = self.obs_dim + self.success_cond_emb_dim

        self.model = get_diffusion_backbone(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            actor_config=actor_cfg,
        ).to(device)

    def _training_obs(self, batch, flatten=True):
        # Get the standard observation data
        nobs = super()._training_obs(batch, flatten=flatten)

        # Get the task embedding
        success = batch["success"].squeeze(-1)
        success_embedding = self.success_embedding(success)

        # With probability p, zero out the success embedding
        B = success_embedding.shape[0]
        blank = torch.rand(B, device=self.device) < self.prob_blank_cond
        success_embedding = success_embedding * ~blank.unsqueeze(-1)

        # Concatenate the task embedding to the observation
        obs_cond = torch.cat((nobs, success_embedding), dim=-1)

        return obs_cond

    # === Inference ===
    def _normalized_action(self, nobs):
        """
        Overwrite the diffusion inference to use the success embedding
        by calling the model with both positive and negative success embeddings
        and without any success embedding.

        The resulting noise prediction will be
        noise_pred = noise_pred_pos - self.guidance_scale * (noise_pred_neg - noise_pred_blank)

        We'll calculate all three in parallel and then split the result into the three parts.
        """
        B = nobs.shape[0]
        # Important! `nobs` needs to be normalized and flattened before passing to this function
        # Initialize action from Guassian noise
        naction = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=self.device,
        )

        # Create 3 success embeddings: positive, negative, and blank
        success_embedding_pos = self.success_embedding(
            torch.tensor(1).to(self.device).repeat(B)
        )
        success_embedding_neg = self.success_embedding(
            torch.tensor(0).to(self.device).repeat(B)
        )
        success_embedding_blank = torch.zeros_like(success_embedding_pos)

        # Concatenate the success embeddings to the observation
        # (B, obs_dim + success_cond_emb_dim)
        obs_cond_pos = torch.cat((nobs, success_embedding_pos), dim=-1)
        obs_cond_neg = torch.cat((nobs, success_embedding_neg), dim=-1)
        obs_cond_blank = torch.cat((nobs, success_embedding_blank), dim=-1)

        # Stack together so that we can calculate all three in parallel
        # (3, B, obs_dim + success_cond_emb_dim)
        obs_cond = torch.stack((obs_cond_pos, obs_cond_neg, obs_cond_blank))

        # Flatten the obs_cond so that it can be passed to the model
        # (3 * B, obs_dim + success_cond_emb_dim)
        obs_cond = obs_cond.view(-1, obs_cond.shape[-1])

        # init scheduler
        self.inference_noise_scheduler.set_timesteps(self.inference_steps)

        for k in self.inference_noise_scheduler.timesteps:
            # Predict the noises for all three success embeddings
            noise_pred_pos, noise_pred_neg, noise_pred_blank = self.model(
                sample=naction.repeat(3, 1, 1),
                timestep=k,
                global_cond=obs_cond,
            ).view(3, B, self.pred_horizon, self.action_dim)

            # Calculate the final noise prediction
            noise_pred = noise_pred_pos - self.guidance_scale * (
                noise_pred_neg - noise_pred_blank
            )

            # inverse diffusion step (remove noise)
            naction = self.inference_noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        return naction
