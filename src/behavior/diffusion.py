from collections import deque
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from src.dataset.normalizer import Normalizer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src.behavior.base import Actor
from src.common.control import RotationMode
from src.models import get_diffusion_backbone, get_encoder
from src.models.unet import ConditionalUnet1D

from ipdb import set_trace as bp  # noqa
from typing import Union


class DiffusionPolicy(Actor):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: Normalizer,
        config: DictConfig,
    ) -> None:
        super().__init__()
        actor_cfg = config.actor
        self.obs_horizon = actor_cfg.obs_horizon
        self.action_dim = (
            10 if config.control.act_rot_repr == RotationMode.rot_6d else 8
        )
        self.pred_horizon = actor_cfg.pred_horizon
        self.action_horizon = actor_cfg.action_horizon

        # A queue of the next actions to be executed in the current horizon
        self.actions = deque(maxlen=self.action_horizon)

        self.inference_steps = actor_cfg.inference_steps
        self.observation_type = config.observation_type
        self.feature_noise = config.regularization.feature_noise
        self.feature_dropout = config.regularization.feature_dropout
        self.feature_layernorm = config.regularization.feature_layernorm
        self.freeze_encoder = freeze_encoder
        self.device = device

        print("Using inference steps:", self.inference_steps)

        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=actor_cfg.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # squared cosine is found to work the best
            beta_schedule=actor_cfg.beta_schedule,
            # clip output to [-1,1] to improve stability
            clip_sample=actor_cfg.clip_sample,
            # our network predicts noise (instead of denoised action)
            prediction_type=actor_cfg.prediction_type,
        )

        self.inference_noise_scheduler = DDIMScheduler(
            num_train_timesteps=actor_cfg.num_diffusion_iters,
            beta_schedule=actor_cfg.beta_schedule,
            clip_sample=actor_cfg.clip_sample,
            prediction_type=actor_cfg.prediction_type,
        )

        # Convert the stats to tensors on the device
        self.normalizer = normalizer.to(device)

        encoder_kwargs = OmegaConf.to_container(config.vision_encoder, resolve=True)
        self.encoder1 = get_encoder(
            encoder_name,
            device=device,
            **encoder_kwargs,
        )
        self.encoder2 = (
            self.encoder1
            if freeze_encoder
            else get_encoder(
                encoder_name,
                device=device,
                **encoder_kwargs,
            )
        )
        self.encoding_dim = self.encoder1.encoding_dim

        if actor_cfg.get("projection_dim") is not None:
            self.encoder1_proj = nn.Linear(
                self.encoding_dim, actor_cfg.projection_dim
            ).to(device)
            self.encoder2_proj = nn.Linear(
                self.encoding_dim, actor_cfg.projection_dim
            ).to(device)
            self.encoding_dim = actor_cfg.projection_dim
        else:
            self.encoder1_proj = nn.Identity()
            self.encoder2_proj = nn.Identity()

        self.flatten_obs = config.actor.diffusion_model.get("flatten_obs", True)
        self.timestep_obs_dim = config.robot_state_dim + 2 * self.encoding_dim
        self.obs_dim = (
            self.timestep_obs_dim * self.obs_horizon
            if self.flatten_obs
            else self.timestep_obs_dim
        )

        self.model = get_diffusion_backbone(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            actor_config=config.actor,
        ).to(device)

        loss_fn_name = (
            config.actor.loss_fn if hasattr(config.actor, "loss_fn") else "MSELoss"
        )
        self.loss_fn = getattr(nn, loss_fn_name)()

    # === Inference ===
    def _normalized_action(self, nobs):
        B = nobs.shape[0]
        # Important! `nobs` needs to be normalized and flattened before passing to this function
        # Initialize action from Guassian noise
        naction = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=self.device,
        )

        # init scheduler
        self.inference_noise_scheduler.set_timesteps(self.inference_steps)

        for k in self.inference_noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.model(sample=naction, timestep=k, global_cond=nobs)

            # inverse diffusion step (remove noise)
            naction = self.inference_noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        return naction

    def _sample_action_pred(self, nobs):
        # Predict normalized action
        # (B, candidates, pred_horizon, action_dim)
        naction = self._normalized_action(nobs)

        # unnormalize action
        # (B, pred_horizon, action_dim)
        action_pred = self.normalizer(naction, "action", forward=False)

        # Add the actions to the queue
        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        actions = deque()
        for i in range(start, end):
            actions.append(action_pred[:, i, :])

        return actions

    @torch.no_grad()
    def action(self, obs: deque):
        # Normalize observations
        nobs = self._normalized_obs(obs, flatten=self.flatten_obs)

        # If the queue is empty, fill it with the predicted actions
        if not self.actions:
            self.actions = self._sample_action_pred(nobs)

        # Return the first action in the queue
        return self.actions.popleft()

    # === Training ===
    def compute_loss(self, batch):
        # State already normalized in the dataset
        obs_cond = self._training_obs(batch, flatten=self.flatten_obs)

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
        loss = self.loss_fn(noise_pred, noise)

        return loss


class MultiTaskDiffusionPolicy(DiffusionPolicy):
    current_task = None

    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: Normalizer,
        config,
    ) -> None:
        super().__init__(
            device=device,
            encoder_name=encoder_name,
            freeze_encoder=freeze_encoder,
            normalizer=normalizer,
            config=config,
        )

        self.task_dim = config.task_dim
        self.task_encoder = nn.Embedding(
            num_embeddings=config.num_tasks,
            embedding_dim=self.task_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
        ).to(device)

        self.obs_dim = self.obs_dim + self.task_dim

        self.model = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=self.obs_dim,
            down_dims=config.down_dims,
        ).to(device)

        self.augment_image = config.augment_image

    def _training_obs(self, batch):
        # Get the standard observation data
        nobs = super()._training_obs(batch, flatten=True)

        # Get the task embedding
        task_idx = batch["task_idx"]
        task_embedding = self.task_encoder(task_idx)

        # Concatenate the task embedding to the observation
        obs_cond = torch.cat((nobs, task_embedding), dim=-1)

        return obs_cond

    def set_task(self, task):
        self.current_task = task

    def _normalized_obs(self, obs: deque):
        assert self.current_task is not None, "Must set current task before calling"

        # Get the standard observation data
        nobs = super()._normalized_obs(obs, flatten=True)
        B = nobs.shape[0]

        # Get the task embedding for the current task and repeat it for the batch size
        task_embedding = self.task_encoder(
            torch.tensor(self.current_task).to(self.device).repeat(B)
        )

        # Concatenate the task embedding to the observation
        obs_cond = torch.cat((nobs, task_embedding), dim=-1)

        return obs_cond


class SuccessGuidedDiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        device: Union[str, torch.device],
        encoder_name: str,
        freeze_encoder: bool,
        normalizer: Normalizer,
        config,
    ) -> None:
        super().__init__(
            device=device,
            encoder_name=encoder_name,
            freeze_encoder=freeze_encoder,
            normalizer=normalizer,
            config=config,
        )

        self.success_cond_dim = 10
        self.success_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.success_cond_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
        ).to(device)

        self.obs_dim = self.obs_dim + self.success_cond_dim

        self.model = get_diffusion_backbone(
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            actor_config=config.actor,
        ).to(device)

    def _training_obs(self, batch, flatten=True):
        # Get the standard observation data
        nobs = super()._training_obs(batch, flatten=flatten)

        # Get the task embedding
        success = batch["success"].squeeze(-1)
        success_embedding = self.success_embedding(success)

        # Concatenate the task embedding to the observation
        obs_cond = torch.cat((nobs, success_embedding), dim=-1)

        return obs_cond

    def _normalized_obs(self, obs: deque, flatten=True):
        # Get the standard observation data
        nobs = super()._normalized_obs(obs, flatten=flatten)
        B = nobs.shape[0]

        # Set the success embedding to true and repeat it for the batch size
        success_embedding = self.success_embedding(
            torch.tensor(1).to(self.device).repeat(B)
        )

        # Concatenate the task embedding to the observation
        obs_cond = torch.cat((nobs, success_embedding), dim=-1)

        return obs_cond
