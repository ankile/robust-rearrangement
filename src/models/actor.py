import torch
from src.data.dataset import normalize_data, unnormalize_data
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ipdb import set_trace as bp
import numpy as np


class Actor:
    def __init__(self, noise_net, config, stats) -> None:
        self.noise_net = noise_net
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        self.obs_horizon = config.obs_horizon
        self.inference_steps = config.inference_steps
        self.device = next(noise_net.parameters()).device
        self.B = config.n_envs

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
    def action(self, obs):
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
            noise_pred = self.noise_net(
                sample=naction, timestep=k, global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        # unnormalize action
        # (B, pred_horizon, action_dim)
        naction = naction[0]
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

    def _normalized_obs(self, obs):
        # Convert agent_pos from obs_horizon x (n_envs, 14) -> (n_envs, obs_horizon, 14)
        agent_pos = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)

        # Convert images from obs_horizon x (n_envs, 224, 224, 3) -> (n_envs, obs_horizon, 224, 224, 3)
        # Also flatten the two first dimensions to get (n_envs * obs_horizon, 224, 224, 3) for the encoder
        img1 = torch.cat([o["color_image1"].unsqueeze(1) for o in obs], dim=1).reshape(
            self.B * self.obs_horizon, 224, 224, 3
        )
        img2 = torch.cat([o["color_image2"].unsqueeze(1) for o in obs], dim=1).reshape(
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
