"""
Elite-noise guided diffusion action generator.

Updates the initial noise distribution using elite samples instead of classifier guidance.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from step_naviexplorer import NaviExplorer


@dataclass
class NoiseConfig:
    lam: float = 1.0          # MPPI temperature: lower = sharper weighting
    alpha: float = 1.0
    var_floor: float = 0.1
    var_cap: Optional[float] = None
    update_variance_threshold: float = 5e-2
    num_steps_update: int = 5


class PriorNaviExplorer(NaviExplorer):
    """
    NaviExplorer that updates the initial-noise distribution
    using elite samples instead of classifier guidance.
    """

    def __init__(
        self,
        dp_checkpoint: str,
        wm_checkpoint: str,
        config_path: str,
        data_folder: str = None,
        data_split_folder: str = None,
        dataset_name: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        noise_config: Optional[NoiseConfig] = None,
        waypoint_spacing: int = 1,
        guidance_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        wm_sampler: str = "ddim",
        wm_ddim_steps: int = 20,
        wm_ddpm_steps: int = 150,
        wm_rollout_mode: str = "oneshot",
    ):
        super().__init__(
            dp_checkpoint=dp_checkpoint,
            wm_checkpoint=wm_checkpoint,
            config_path=config_path,
            data_folder=data_folder,
            data_split_folder=data_split_folder,
            dataset_name=dataset_name,
            device=device,
            waypoint_spacing=waypoint_spacing,
            # Forwarded so this subclass can also fall back to the legacy world-model path:
            # wm_sampler='ddpm', wm_rollout_mode='autoregressive'.
            wm_sampler=wm_sampler,
            wm_ddim_steps=wm_ddim_steps,
            wm_ddpm_steps=wm_ddpm_steps,
            wm_rollout_mode=wm_rollout_mode,
        )
        self.guidance_loss_fn = guidance_loss_fn
        self._goal_yaw = None
        self.noise_config = noise_config or NoiseConfig()
        self._noise_mu = None
        self._noise_var = None
        self.update_counter = 0
    
    def reset_noise_distribution(self):
        """Reset the noise distribution to standard normal for a new rollout sequence."""
        self._noise_mu = None
        self._noise_var = None

    def adjust_noise_distribution(self):
        # We will keep the mean as is but increase variance to encourage more exploration
        if self._noise_var is not None:
            self._noise_var = self._noise_var * 1.5

    def _sample_noise(
        self,
        num_samples: int,
        device: torch.device,
        action_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        len_traj = self.config["len_traj_pred"]
        d = int(len_traj * action_dim)
        if self._noise_mu is None:
            self._noise_mu = torch.zeros(d, device=device, dtype=dtype)
            self._noise_var = torch.ones(d, device=device, dtype=dtype)
        eps = torch.randn((num_samples, d), device=device, dtype=dtype)
        z0_flat = self._noise_mu + torch.sqrt(self._noise_var) * eps
        return z0_flat.view(num_samples, len_traj, action_dim)

    def _update_noise_distribution(
        self,
        z0: torch.Tensor,
        score: torch.Tensor,
    ) -> None:
        
        score = score.detach().to(z0.device)

        # # MPPI exponential weighting: w_i = exp((s_i - max(s)) / lambda)
        # # Subtract max for numerical stability before exponentiation
        # z0_flat = z0.reshape(z0.shape[0], -1)
        # shifted = (score - score.max()) / self.noise_config.lam
        # w = torch.exp(shifted)
        # w = w / w.sum()  # normalize to sum to 1

        # # Weighted mean and variance
        # mu_w = (w.unsqueeze(1) * z0_flat).sum(dim=0).float()
        # diff = z0_flat - mu_w.unsqueeze(0)
        # var_w = (w.unsqueeze(1) * diff ** 2).sum(dim=0).float()

        # Choose top action and use its values as mu_w and var_w
        top_idx = torch.argmax(score)
        mu_w = z0[top_idx].reshape(-1)
        var_w = torch.ones_like(mu_w) * 0.65

        self._noise_mu = (1 - self.noise_config.alpha) * self._noise_mu + self.noise_config.alpha * mu_w
        self._noise_var = (1 - self.noise_config.alpha) * self._noise_var + self.noise_config.alpha * var_w
        self._noise_var = self._noise_var.clamp_min(self.noise_config.var_floor)
        if self.noise_config.var_cap is not None:
            self._noise_var = self._noise_var.clamp_max(self.noise_config.var_cap)
        
        print(
            f"[noise update] lam={self.noise_config.lam}, "
            f"mu norm: {self._noise_mu.norm().item():.6f}, "
            f"var mean: {self._noise_var.mean().item():.6f}"
        )
        return mu_w, score[top_idx]
    
    def autoregressive_rollout(
        self,
        obs_images: torch.Tensor,
        long_horizon_context: Union[torch.Tensor, None],
        # Visualization only (fixed)
        goal_image: torch.Tensor = None,
        goal_yaw: torch.Tensor = None,
        action_label: torch.Tensor = None,
        goal_pos: torch.Tensor = None,
        # Config
        rollout_steps: int = 8,
        visualize: bool = True,
        save_dir: str = "rollout_output",
        mask_type: str = "goal_mask",
        **kwargs,
    ) -> Dict:
        """
        Complete pipeline with noise-guided diffusion.
        
        Args:
            obs_images: Short context images for local planning (C*context_size, H, W) in [0,1]
            obs_latents: Short context latents for local planning
            long_horizon_context: Accumulated latents for dissimilarity scoring
            robot_position: Current robot position (x, y) tensor
            robot_yaw: Current robot yaw angle
            goal_image: Goal image (optional, for visualization only)
            goal_yaw: Goal yaw angle (optional, for visualization only)
            action_label: Ground truth action labels (optional, for visualization)
            goal_pos: Goal position (optional, for visualization)
            num_samples: Number of diverse action sequences to generate
            num_outputs: Number of output trajectories for visualization
            rollout_steps: Number of autoregressive steps
            visualize: Whether to visualize results
            save_dir: Directory to save visualizations
            mask_type: Type of mask to apply during action sequence generation
        
        Returns:
            Dictionary with action sequences, generated future states, and all state needed for next rollout
        """
        print("\n" + "="*60)

        # Preprocess images to generate latents
        obs_images = obs_images * 2 - 1  # Scale to [-1, 1]
        obs_images = obs_images.to(self.device)  # (context_size, C, H, W)
        if goal_image is not None:
            goal_image = goal_image * 2 - 1  # Scale to [-1, 1]
            goal_image = goal_image.to(self.device)  # (3, H, W)
        else:
            goal_image = obs_images[0]

        obs_latents = self._vae_encode(obs_images).unsqueeze(0)  # (1, context_size, D, H, W)
        goal_latent = self._vae_encode(goal_image.unsqueeze(0)) if goal_image is not None else None

        # Update long horizon context with current observation latents
        if long_horizon_context is None:
                long_horizon_context = obs_latents[0,-1:]  # Use last latent only
        elif kwargs.get("lhc_flag", False):
                long_horizon_context = torch.cat([
                    long_horizon_context,
                    obs_latents[0,-1:]
                ], dim=0)

        action_sequences = None
        future_latents = None
        score = None
        num_samples = kwargs.get("num_samples", 20)
        num_buckets = kwargs.get("num_outputs", num_samples)

        # First pass: sample actions from the current noise distribution.
        results = self.generate_action_sequences(
            obs_latents=obs_latents,
            goal_latent=goal_latent,
            num_samples=num_samples,
            num_buckets=num_buckets,
            mask_type=mask_type,
            action_dim=3 if self.config["learn_angle"] else 2,
        )

        action_sequences = results['action_sequences']
        old_action_sequences = action_sequences.clone()
        initial_noise = results['initial_noise']
        action_variance = torch.std(action_sequences, dim=0)
        mean_action_variance = action_variance.mean().item()
        should_update_noise = torch.any(
            action_variance > self.noise_config.update_variance_threshold
        ).item()
        print(f"Action variance: {mean_action_variance:.4f}")

        if should_update_noise:
            self.update_counter += 1

            with torch.no_grad():
                future_latents = self.generate_future_states_batched(
                    obs_latents=obs_latents,
                    action_sequences=action_sequences,
                    goal_yaw=goal_yaw if goal_yaw is not None else torch.tensor(0.0, device=self.device),
                    num_steps=rollout_steps,
                )

            dissimilarity_scores = self._fastnn_dissimilarity_score(
                context_latents=long_horizon_context,
                future_latents=future_latents,
            )
            score = torch.from_numpy(dissimilarity_scores).to(self.device)

            _, weighted_score = self._update_noise_distribution(initial_noise, score)
            # print(
            #     f"[noise] pre-update score mean={score.mean().item():.6f} "
            #     f"weighted={weighted_score.item():.6f} "
            #     f"min={score.min().item():.6f} max={score.max().item():.6f}"
            # )

            results = self.generate_action_sequences(
                obs_latents=obs_latents,
                goal_latent=goal_latent,
                num_samples=1,
                num_buckets=1,
                mask_type=mask_type,
                action_dim=3 if self.config["learn_angle"] else 2,
            )
            action_sequences = results['action_sequences']
            best_action_idx = 0
            best_score = float(weighted_score.item())
            # future_latents = None
        else:
            if self.update_counter > 0:
                self.update_counter += 1
            score = torch.zeros((action_sequences.shape[0],), device=self.device)
            best_score = 0.0
            best_action_idx = np.random.randint(0, action_sequences.shape[0])

        if self.update_counter == self.noise_config.num_steps_update:
            self.reset_noise_distribution()
            print(f"Resetting noise distribution at update step {self.update_counter}.")
            self.update_counter = 0
        
        action_sequences_denorm = self._denorm(action_sequences)
        best_action_sequence = action_sequences_denorm[best_action_idx]

        yaw_deltas = self._del_yaw_from_deltas(best_action_sequence)
        yaw_values = yaw_deltas.cumsum(dim=0)

        if visualize and should_update_noise:
            os.makedirs(save_dir, exist_ok=True)
            viz_obs_images = self._to_display_images(obs_images, resize_hw=(240, 320))
            old_action_sequences_denorm = self._denorm(old_action_sequences)
            action_sequences_np = old_action_sequences_denorm.cpu().numpy()
            action_sequences_np = action_sequences_np[..., :2]  # Only x,y for visualization
            action_label_np = action_label.cpu().numpy() if action_label is not None else None
            action_label_np = action_label_np[..., :2] if action_label_np is not None else None
            dissimilarity_scores_values = score.detach().cpu().tolist()
            goal_pos_np = goal_pos.cpu().numpy() if goal_pos is not None else None
            print(f"Actions: {action_sequences_np.shape}, Scores: {len(dissimilarity_scores_values)}")
            self._visualize_action_sequences(
                curr_image=viz_obs_images[-1],
                action_sequences=action_sequences_np,
                batch_action_label=action_label_np,
                batch_goal_pos=goal_pos_np,
                dissimilarity_scores=dissimilarity_scores_values,
                save_dir=save_dir,
                viz_type=["graph", "image"],
                **kwargs,
            )

            if future_latents is not None:
                future_states = []
                for i in range(future_latents.shape[0]):
                    # Get the final predicted state in the rollout
                    state = self._vae_decode(future_latents[i][-1:]) 
                    state = self._to_display_images(state, resize_hw=(240, 320))
                    # Extract from batch dim and transpose from (C, H, W) to (H, W, C)
                    state = np.transpose(state[0], (1, 2, 0))
                    state = Image.fromarray((state * 255).astype(np.uint8))
                    state.save(os.path.join(save_dir, f"future_state_{i}.png"))
                    future_states.append(state)
                future_states = np.stack(future_states)
                # save states as np arrays for later visualization
                # self._visualize_states(
                #     save_dir=save_dir,
                #     states=future_states,
                #     save_name="future_states.png",
                # )

            self._visualize_states(
                save_dir=save_dir,
                states=viz_obs_images,
                save_name="context_states.png",
            )

        combined_results = {
            'action_sequences': best_action_sequence,
            'yaw_values': yaw_values,
            'future_latents': future_latents,
            'long_horizon_context': long_horizon_context,
            'dissimilarity_score': best_score,
            'noise_mu': self._noise_mu.detach().clone() if self._noise_mu is not None else None,
            'noise_var': self._noise_var.detach().clone() if self._noise_var is not None else None,
        }

        print("="*60)

        return combined_results
