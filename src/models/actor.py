import torch
from src.data.dataset import normalize_data, unnormalize_data
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import ipdb


class Actor:
    def __init__(self, noise_net, config, stats) -> None:
        self.noise_net = noise_net
        self.config = config
        self.stats = stats
        self.device = next(noise_net.parameters()).device
        self.B = 1

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )

    def _normalized_obs(self, obs):
        raise NotImplementedError

    @torch.no_grad()
    def action(self, obs):
        obs_cond = self._normalized_obs(obs)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (self.B, self.config.pred_horizon, self.config.action_dim),
            device=self.device,
        )
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(self.config.inference_steps)

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
        naction = naction.detach().to("cpu").numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])

        return action_pred


class StateActor(Actor):
    def __init__(self, noise_net, config, stats) -> None:
        raise NotImplementedError


class ImageActor(Actor):
    def __init__(self, noise_net, encoder, config, stats) -> None:
        super().__init__(noise_net, config, stats)
        self.encoder = encoder

    def _normalized_obs(self, obs):
        agent_pos = torch.cat([o["robot_state"].unsqueeze(1) for o in obs], dim=1)
        nobs = normalize_data(agent_pos.cpu(), stats=self.stats["agent_pos"]).to(
            self.device
        )
        img1 = torch.cat([o["color_image1"] for o in obs], dim=0).transpose(3, 1)
        img2 = torch.cat([o["color_image2"] for o in obs], dim=0).transpose(3, 1)

        feature1 = self.encoder(img1).reshape(self.B, self.config.obs_horizon, -1)
        feature2 = self.encoder(img2).reshape(self.B, self.config.obs_horizon, -1)

        nobs = torch.cat([nobs, feature1, feature2], dim=-1).flatten(start_dim=1)

        return nobs
