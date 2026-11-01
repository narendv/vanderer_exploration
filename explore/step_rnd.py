"""
RND-augmented NaviExplorer.

Uses the base NaviExplorer rollout implementation and plugs in
an RND-based novelty loss for trajectory scoring.
"""

import os
import time
import yaml
from typing import List, Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import torch.nn.functional as F
import re
import sys

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.training.train_utils import get_action, ACTION_STATS
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.visualizing.action_utils import plot_trajs_and_points, plot_trajs_and_points_on_image
from nwm.models import CDiT_models
from expnav.data.vint_dataset import ViNT_Dataset
from nwm.diffusion import create_diffusion
from rnd_reward import ConvEncoder

class NaviExplorer:
    """NaviExplorer variant with RND novelty scoring."""

    def __init__(
        self,
        dp_checkpoint: str,
        wm_checkpoint: str,
        config_path: str,
        dataset_name: str = None,
        waypoint_spacing: int = 4,
        rnd_train_iters: int = 10,
        rnd_buffer_size: int = 256,
        rnd_batch_size: int = 32,
        rnd_lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        wm_sampler: str = "ddim",
        wm_ddim_steps: int = 20,
        wm_ddpm_steps: int = 150,
        wm_rollout_mode: str = "oneshot",
    ):
        """
        wm_sampler: World-model sampler, 'ddim' (fast, default) or 'ddpm' (legacy).
        wm_ddim_steps: Denoising steps when wm_sampler == 'ddim'.
        wm_ddpm_steps: Denoising steps when wm_sampler == 'ddpm' (150 = legacy behaviour).
        wm_rollout_mode: 'oneshot' predicts the final state in a single jump using the
            cumulative action and a scaled rel_t. 'autoregressive' is the legacy
            step-by-step rollout. Use 'autoregressive' to fall back, or when you need
            the intermediate frames for visualisation.
        """
        self.device = torch.device(device)
        self.config = self._load_config(config_path)
        self.config['waypoint_spacing'] = waypoint_spacing
        assert wm_sampler in ("ddim", "ddpm"), f"unknown wm_sampler {wm_sampler}"
        assert wm_rollout_mode in ("oneshot", "autoregressive"), f"unknown wm_rollout_mode {wm_rollout_mode}"
        self.wm_sampler = wm_sampler
        self.wm_ddim_steps = wm_ddim_steps
        self.wm_ddpm_steps = wm_ddpm_steps
        self.wm_rollout_mode = wm_rollout_mode
        self._action_stats_t = None
        self.wm_checkpoint = wm_checkpoint
        # Load model architecture
        self.model, self.world_model = self._build_model()
        self.dataset_name = dataset_name
        
        # Load checkpoint and EMA model
        self._load_checkpoint(dp_checkpoint)
        
        # Setup noise scheduler for diffusion
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )


        self._denorm = get_action

        self.rnd_train_iters = rnd_train_iters
        self.rnd_buffer_size = rnd_buffer_size
        self.rnd_batch_size = rnd_batch_size
        self.rnd_latent_buffer: List[torch.Tensor] = []
        self.rnd_train_loss: float = 0.0

        latent_size = self.config["image_size"][0] // 8
        rnd_obs_shape = (4, latent_size, latent_size)
        self.rnd_target = ConvEncoder(obs_shape=rnd_obs_shape,
                                      hidden_size_list=[32, 64, 64, 128],
                                      activation=torch.nn.ReLU(),
                                      kernel_size=[4, 3, 3],
                                      stride=[4, 2, 1],
                                      padding=None,
                                      layer_norm=True,
                                      norm_type='GN'
                        ).to(self.device)
        
        self.rnd_predictor = ConvEncoder(obs_shape=rnd_obs_shape,
                                      hidden_size_list=[32, 64, 64, 128],
                                      activation=torch.nn.ReLU(),
                                      kernel_size=[4, 3, 3],
                                      stride=[4, 2, 1],
                                      padding=None,
                                      layer_norm=True,
                                      norm_type='GN'
                        ).to(self.device)
        for param in self.rnd_target.parameters():
            param.requires_grad = False
        self.rnd_target.eval()
        self.rnd_optimizer = torch.optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _build_model(self) -> NoMaD:
        """Build the NoMaD model with diffusion components."""
        config = self.config
        
        # Vision encoder
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                obs_encoder=config["obs_encoder"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                use_cache_encodings=True,
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        # Noise prediction network (diffusion model)
        noise_pred_net = ConditionalUnet1D(
            input_dim=3 if config["learn_angle"] else 2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        
        # Distance prediction network
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        # Combine into NoMaD model
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        # Set model to eval mode
        model.eval()
        model.to(self.device)
        
        # Extract model name from checkpoint path
        match = re.search(r'cdit_(\w+)', self.wm_checkpoint)
        model_name = f"CDiT-{match.group(1)[:1].upper()}/2"
        cdit = CDiT_models[model_name](context_size=4, input_size=28, in_channels=4)
        ckp = torch.load(self.wm_checkpoint, map_location='cpu', weights_only=False)
        cdit.load_state_dict(ckp["ema"], strict=True)

        # cdit.eval()
        cdit.to(self.device)
        for param in cdit.parameters():
            param.requires_grad = False
        cdit = torch.compile(cdit)
        # Both schedules are built so the sampler can be switched at runtime without a reload.
        diffusion = {
            'ddpm': create_diffusion(str(self.wm_ddpm_steps)),
            'ddim': create_diffusion(str(self.wm_ddim_steps)),
        }
        print(f"World model: sampler={self.wm_sampler} "
              f"({self.wm_ddim_steps if self.wm_sampler == 'ddim' else self.wm_ddpm_steps} steps), "
              f"rollout_mode={self.wm_rollout_mode}")
        # for param in vision_encoder.obs_encoder.parameters():
        #     param.requires_grad = False
        model_lst = (cdit, diffusion, vision_encoder.obs_encoder)
        
        return model, model_lst
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint and return EMA model if available."""
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # Handle DataParallel + Optimized models
        if "_orig_mod.module." in list(checkpoint.keys())[0]:
            new_checkpoint = {}
            for k, v in checkpoint.items():
                new_key = k.replace("_orig_mod.module.", "")
                new_checkpoint[new_key] = v
            checkpoint = new_checkpoint

        # Handle Optimized models
        elif "_orig_mod." in list(checkpoint.keys())[0]:
            new_checkpoint = {}
            for k, v in checkpoint.items():
                new_key = k.replace("_orig_mod.", "")
                new_checkpoint[new_key] = v
            checkpoint = new_checkpoint
        
        self.model.load_state_dict(checkpoint, strict=True)
        
        # Load EMA model if available
        ema_model = None
        ema_checkpoint_path = checkpoint_path.replace("orig", "ema")
        if os.path.exists(ema_checkpoint_path):
            from diffusers.training_utils import EMAModel
            ema_state_dict = torch.load(ema_checkpoint_path, map_location=self.device, weights_only=False)
            ema_model = EMAModel(parameters=self.model.parameters())
            ema_model.load_state_dict(ema_state_dict)
            ema_model.copy_to(self.model.parameters())
            print(f"EMA model weights loaded successfully from {ema_checkpoint_path}")
        else:
            print(f"No EMA model found at {ema_checkpoint_path}, using original model")
        self.model.eval()
    
    def _vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using the VAE encoder from the world model."""
        _, _, vae = self.world_model
        assert images.dim() == 4, "Expected images to be a batch of 4D tensors (B, C, H, W)"
        with torch.no_grad():
            latents = vae.encode(images.to(self.device)).latent_dist.mean
        return latents
    
    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to images using the VAE decoder from the world model."""
        _, _, vae = self.world_model
        with torch.no_grad():
            decoded_images = vae.decode(latents.to(self.device)).sample
        decoded_images = torch.clamp(decoded_images, -1, 1)  # Ensure outputs are in valid range
        return decoded_images

    def _to_display_images(
        self,
        images: torch.Tensor,
        resize_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Convert model-space images in [-1, 1] to numpy images in [0, 1]."""
        display_images = images.detach().to(dtype=torch.float32)
        if resize_hw is not None:
            display_images = TF.resize(display_images, resize_hw)
        display_images = ((display_images + 1.0) / 2.0).clamp(0.0, 1.0)
        return display_images.cpu().numpy()
    
    def _encode_latents(
        self,
        obs_latents: torch.Tensor,
        goal_latent: torch.Tensor,
        mask_type: str,
    ) -> torch.Tensor:
        """Encode observation and goal latents using vision encoder."""
        B = obs_latents.shape[0]
        if mask_type == "no_mask":
            goal_mask_tensor = torch.zeros((B,)).long().to(self.device)
        else:
            goal_mask_tensor = torch.ones((B,)).long().to(self.device)
        
        with torch.no_grad():
            encoded = self.model(
                "vision_encoder",
                obs_img=obs_latents,
                goal_img=goal_latent,
                input_goal_mask=goal_mask_tensor,
            )
        
        return encoded
    
    def _integrate_deltas_to_actions(self, deltas: torch.Tensor) -> torch.Tensor:
        """Convert action deltas to absolute actions by cumulative sum."""
        actions = torch.cumsum(deltas, axis=1)
        return actions
    
    def _cumulative_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Collapse per-step policy deltas into one cumulative action for a one-shot jump.

        NoMaD emits *normalized per-step deltas*; the world model expects the goal pose
        relative to the current frame, normalized the same way. normalize_data is affine
        with a non-zero offset, so naively summing the normalized values would accumulate
        that offset T times. Un-normalize, sum, then re-normalize.

        actions: (N, T, 3) -> (N, 1, 3)
        """
        if self._action_stats_t is None:
            self._action_stats_t = (
                torch.as_tensor(ACTION_STATS['min'], dtype=torch.float32, device=self.device),
                torch.as_tensor(ACTION_STATS['max'], dtype=torch.float32, device=self.device),
            )
        lo, hi = self._action_stats_t
        a = actions.to(self.device).float()
        xy = (a[..., :2] + 1) / 2 * (hi - lo) + lo   # -> raw per-step displacements
        xy = xy.sum(dim=1)                            # cumulative displacement (N, 2)
        xy = (xy - lo) / (hi - lo) * 2 - 1            # -> normalized goal position
        yaw = a[..., 2:3].sum(dim=1)                  # total yaw change (raw radians)
        return torch.cat([xy, yaw], dim=-1).unsqueeze(1)

    def _del_yaw_from_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        thetas = torch.atan2(deltas[..., 1], deltas[..., 0])
        pi = torch.acos(torch.zeros(1)).item() * 2
        thetas = thetas - 2 * pi * torch.floor((thetas + pi) / (2 * pi))
        thetas = torch.cat([torch.tensor([0], device=self.device), thetas], dim=-1)
        delta_theta = thetas[1:] - thetas[:-1]
        return delta_theta.unsqueeze(-1)

    def _sample_noise(
        self,
        num_samples: int,
        device: torch.device,
        action_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(
            (num_samples, self.config["len_traj_pred"], action_dim),
            device=device,
            dtype=dtype,
        )

    def _farthest_point_sampling(
        self, 
        action_sequences: torch.Tensor,
        num_buckets: int = 3
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        samples: (num_samples, trajectory_length, 2)
        num_outputs: Number of distinct samples to return
        """
        # Flatten trajectories to (num_samples, trajectory_length * 2)
        N = action_sequences.shape[0]
        data = action_sequences.reshape(N, -1)

        # Initialize: pick the first sample
        selected_indices = [0]
        distances = torch.full((N,), float("inf"), device=data.device, dtype=data.dtype)

        for _ in range(1, num_buckets):
            # Last added point
            latest_point = data[selected_indices[-1]]

            # Update distances: min distance from each point to the set of selected points
            current_dist = torch.norm(data - latest_point, dim=1)
            distances = torch.minimum(distances, current_dist)

            # Avoid selecting already-selected points again
            distances[selected_indices] = 0.0

            # Pick the point that is furthest from its nearest neighbor in the selected set
            next_index = int(torch.argmax(distances).item())
            selected_indices.append(next_index)

        selected_actions = action_sequences[selected_indices]

        return selected_actions, selected_indices

    def _append_obs_latents_to_buffer(self, obs_latent: torch.Tensor) -> int:
        """Append encoded observation latents to a fixed-size replay buffer."""
        obs_latents = obs_latent.detach().to(dtype=torch.float32, device="cpu")
        if obs_latents.dim() == 4:
            obs_latents = obs_latents[0]

        self.rnd_latent_buffer.append(obs_latent.clone())

        if len(self.rnd_latent_buffer) > self.rnd_buffer_size:
            self.rnd_latent_buffer = self.rnd_latent_buffer[-self.rnd_buffer_size :]

        return len(self.rnd_latent_buffer)

    def _train_rnd_predictor_from_buffer(self):
        """Train RND predictor from random minibatches sampled from the latent buffer."""
        num_iters = self.rnd_train_iters

        batch_size = min(self.rnd_batch_size, self.rnd_buffer_size)
        self.rnd_predictor.train()
        final_loss = 0.0
        for _ in range(num_iters):
            indices = torch.randint(0, self.rnd_buffer_size, (batch_size,))
            batch = torch.stack([self.rnd_latent_buffer[int(i)] for i in indices], dim=0).to(self.device)
            with torch.no_grad():
                target_feat = self.rnd_target(batch)
            pred_feat = self.rnd_predictor(batch)
            loss = F.mse_loss(pred_feat, target_feat)
            self.rnd_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.rnd_optimizer.step()
            final_loss = float(loss.item())

        self.rnd_train_loss = final_loss

    def _rnd_novelty_score(self, future_latents: torch.Tensor) -> np.ndarray:
        """Compute per-sample RND novelty score from the final predicted latent state."""
        final_states = future_latents[:, -1].to(self.device)
        self.rnd_predictor.eval()
        with torch.no_grad():
            target_feat = self.rnd_target(final_states)
            pred_feat = self.rnd_predictor(final_states)
            scores = F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=1)
        return scores.detach().cpu().numpy()

    def generate_action_sequences(
        self,
        obs_latents: torch.Tensor,
        goal_latent: torch.Tensor,
        num_samples: int = 20,
        num_buckets: Union[int, None] = 3,
        mask_type: str = "no_mask",
        action_dim: int = 2,
    ) -> Dict:
        obs_cond = self._encode_latents(
            obs_latents=obs_latents,
            goal_latent=goal_latent,
            mask_type=mask_type,
        )
        obs_cond = obs_cond.flatten(start_dim=1)

        with torch.no_grad():
            obs_cond = obs_cond.repeat(num_samples, 1)
            self.noise_scheduler.set_timesteps(self.config["num_diffusion_iters"])

        naction = None
        
        z0 = self._sample_noise(
                    num_samples=num_samples,
                    device=self.device,
                    action_dim=action_dim,
                    dtype=torch.float32,
                )
        initial_noise = z0.clone()
        naction = z0

        for k in self.noise_scheduler.timesteps[:]:
            with torch.no_grad():
                noise_pred = self.model(
                    'noise_pred_net',
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                ).prev_sample

        result = {}

        result['all_action_sequences'] = naction
        if num_buckets < num_samples:
            selected_actions, selected_indices = self._farthest_point_sampling(
                naction, num_buckets=num_buckets
            )
            result['action_sequences'] = selected_actions
            result['initial_noise'] = initial_noise[selected_indices]
            # result['predicted_distances'] = predicted_distances[selected_indices]
            result['selected_indices'] = selected_indices
        else:
            result['initial_noise'] = initial_noise
            result['action_sequences'] = naction
            # result['predicted_distances'] = predicted_distances

        return result

    def generate_future_states_batched(
        self,
        obs_latents: torch.Tensor,
        action_sequences: torch.Tensor,
        goal_yaw: torch.Tensor,
        num_steps: int = None,
    ) -> Dict:
        """
        Autoregressively generate future states by applying actions to a world model.
        BATCHED VERSION: Processes all samples in parallel through the world model.
        
        Args:
            obs_latents: Observation latents tensor
            action_sequences: Generated action sequences of shape (num_samples, trajectory_length, 2)
            goal_yaw: Goal yaw angle
            num_steps: Number of autoregressive steps to generate. If None, uses trajectory_length
        
        Returns:
            Dictionary with:
                - 'future_states': Generated future states/images
                - 'action_sequences': The action sequences used
                - 'trajectory_length': Number of states generated
        """
        if num_steps is None:
            num_steps = action_sequences.shape[1]
        
        num_samples = action_sequences.shape[0]
        
        # Prepare actions for all samples: add yaw deltas
        all_sample_actions = []
        for sample_idx in range(num_samples):
            sample_actions = action_sequences[sample_idx].float()  # (traj_len, 2)
            # assert sample_actions.shape[-1] == 3, "Expected action sequences to have shape (num_samples, trajectory_length, 3)"
            if sample_actions.shape[-1] == 2:
                yaw_deltas = self._del_yaw_from_deltas(sample_actions)
                sample_actions = torch.cat([sample_actions, yaw_deltas], dim=-1)  # (traj_len, 3)
            all_sample_actions.append(sample_actions)
        
        all_sample_actions = torch.stack(all_sample_actions)  # (num_samples, traj_len, 3)
        
        # Initialize contexts for all samples - all start with same observation
        # Repeat for all samples: (num_samples, context_size, 3, H, W)
        batch_context = obs_latents.clone().repeat(num_samples, 1, 1, 1, 1)
        
        all_future_latents = []
        start_time = time.time()

        if self.wm_rollout_mode == 'oneshot':
            # Predict the state num_steps ahead in ONE forward instead of num_steps.
            # The RND scorer only reads future_latents[:, -1], so the intermediate frames
            # are not needed unless you are visualising. Conditioning is the cumulative
            # action over the whole horizon, with rel_t scaled to match.
            cum_actions = self._cumulative_action(all_sample_actions[:, :num_steps, :])  # (N,1,3)
            rel_t = torch.full(
                (num_samples,),
                num_steps * self.config['waypoint_spacing'] / 128.,
                device=self.device,
            )
            with torch.no_grad():
                final_latents = self._world_model_forward(
                    batch_context,
                    cum_actions,
                    num_timesteps=num_steps * self.config['waypoint_spacing'],
                    rel_t=rel_t,
                )
            future_latents = final_latents.detach().unsqueeze(1)  # (N, 1, 4, h, w)
            print(f"One-shot rollout for {num_samples} samples "
                  f"(+{num_steps * self.config['waypoint_spacing']} frames): "
                  f"{time.time() - start_time:.2f} seconds")
            return future_latents

        # Autoregressive rollout - batched across all samples
        for step in range(num_steps):
            batch_action_deltas = all_sample_actions[:, step, :].unsqueeze(1)  # (num_samples, 1, 3)
            # Forward pass through world model for all samples at once
            with torch.no_grad():
                batch_output_latents = self._world_model_forward(
                    batch_context,  # (num_samples, context_size, 3, H, W)
                    batch_action_deltas,  # (num_samples, 1, 3)
                    num_timesteps=self.config['waypoint_spacing'],
                )  # latents: (num_samples, 4, h, w)
            
            # Store generated latents
            all_future_latents.append(batch_output_latents.detach())
            
            # Update encoded contexts for all samples: slide window and add new latent
            batch_context = torch.cat(
                [batch_context[:, 1:], batch_output_latents.unsqueeze(1)],
                dim=1,
            )
        end_time = time.time()
        
        future_latents = torch.stack(all_future_latents, dim=1)
        
        return future_latents
    
    def _world_model_forward(
        self, 
        obs_latents,
        curr_delta,
        num_timesteps,
        num_goals=1,
        rel_t=None,
        sampler=None,
    ):
        model, diffusion, vae = self.world_model
        x = obs_latents.to(self.device)
        y = curr_delta.to(self.device)
        num_cond = self.config["context_size"]+1
        latent_size = self.config["image_size"][0] // 8

        sampler = self.wm_sampler if sampler is None else sampler
        dif = diffusion[sampler]

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]

            if rel_t is None:
                rel_t = (torch.ones(B)* (1. / 128.)).to(self.device)
                rel_t *= num_timesteps

            # Scale latents and prepare conditioning states
            x = x * 0.18215
            x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)

            z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=self.device)
            y = y.flatten(0, 1)
            model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)

            if sampler == 'ddim':
                # eta=0 -> deterministic; matches DDPM quality at ~1/8 the steps.
                samples = dif.ddim_sample_loop(
                    model.forward, z.shape, noise=z, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, device=self.device, eta=0.0
                )
            else:
                samples = dif.p_sample_loop(
                    model.forward, z.shape, z, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, device=self.device
                )

            encoded_samples = (samples/ 0.18215).to(dtype=torch.float32)
        return encoded_samples
    
    def autoregressive_rollout(
        self,
        obs_images: torch.Tensor,
        # Visualization only (fixed)
        goal_image: torch.Tensor = None,
        goal_yaw: torch.Tensor = None,
        action_label: torch.Tensor = None,
        goal_pos: torch.Tensor = None,
        # Config
        num_samples: int = 20,
        num_outputs: int = 5,
        rollout_steps: int = 8,
        visualize: bool = True,
        save_dir: str = "rollout_output",
        mask_type: str = "goal_mask",
        **kwargs,
    ) -> Dict:
        """
        Complete pipeline: generate actions from provided images, then autoregressively rollout.
        Now accepts data externally instead of loading from dataset.
        
        Args:
            obs_images: Observation images tensor (context_size, C, H, W)
            goal_image: Goal (3, H, W)
            long_horizon_latents: Mapped latents for dissimilarity comparison
            goal_yaw: Goal yaw angle
            action_label: Ground truth action labels (optional, for visualization)
            goal_pos: Goal position (optional, for visualization)
            context_latents: Accumulated context latents from previous rollouts (optional)
            num_samples: Number of diverse action sequences to generate
            num_outputs: Number of output trajectories after bucketing
            visualize: Whether to visualize results
            save_dir: Directory to save visualizations
            mask_type: Type of mask to apply during action sequence generation
        Returns:
            Dictionary with action sequences, generated future states, and all state needed for next rollout
        """
        
        # # Get goal position for visualization if available
        # goal_pos_np = goal_pos.numpy() if goal_pos is not None else None
        # action_label_np = action_label.numpy() if action_label is not None else None
        
        print("\n" + "="*60)
        selection_threshold = 5e-2  # Threshold for action variance to decide if dissimilarity-based selection is needed
        # Preprocess images to generate latents
        obs_images = obs_images * 2 - 1  # Scale to [-1, 1]
        obs_images = obs_images.to(self.device) # (context_size, C, H, W)
        if goal_image is not None:
            goal_image = goal_image * 2 - 1  # Scale to [-1, 1]
            goal_image = goal_image.to(self.device)  # (3, H, W)
        else:
            goal_image = obs_images[0]  # Dummy goal
        
        obs_latents = self._vae_encode(obs_images).unsqueeze(0)  # (1, context_size, D, H, W)
        goal_latent = self._vae_encode(goal_image.unsqueeze(0)) if goal_image is not None else None  # (1, D, H, W)
        action_sequences = None

        # Update RND model
        buffer_length = self._append_obs_latents_to_buffer(obs_latents[0,-1])
        if buffer_length == self.rnd_buffer_size:
            self._train_rnd_predictor_from_buffer()
            self.rnd_latent_buffer = []

        # Generate action sequences using the loaded latents
        results = self.generate_action_sequences(
            obs_latents=obs_latents,
            goal_latent=goal_latent,
            num_samples=num_samples,
            num_buckets=num_outputs,
            mask_type=mask_type,
            action_dim=3 if self.config["learn_angle"] else 2,
        )
        
        action_sequences = results['action_sequences']
        

        action_variance = torch.std(action_sequences, dim=0)
        print(f"Action variance: {action_variance.mean().item():.4f}")
        do_selection = torch.any(action_variance > selection_threshold)
        if do_selection:
            future_latents = self.generate_future_states_batched(
                                obs_latents=obs_latents,
                                action_sequences=action_sequences,
                                num_steps=rollout_steps,
                                goal_yaw=goal_yaw,
                            )

            dissimilarity_scores = self._rnd_novelty_score(future_latents)
            best_action_idx = np.argmax(dissimilarity_scores)
            best_score = dissimilarity_scores[best_action_idx]
        else:
            # Randomly choose best action index for demonstration
            dissimilarity_scores = torch.zeros((num_outputs,))
            best_score = 0.0
            future_latents = torch.zeros((num_outputs, rollout_steps, 4, 28, 28))
            best_action_idx = np.random.randint(0, action_sequences.shape[0])

        action_sequences_denorm = get_action(action_sequences)
        best_action_sequence = action_sequences_denorm[best_action_idx]  # (trajectory_length, 2)
        
        # Visualization
        if visualize and do_selection:
            os.makedirs(save_dir, exist_ok=True)
            # Convert images to numpy for visualization.
            obs_images = self._to_display_images(obs_images, resize_hw=(240, 320))

            action_sequences_np = action_sequences_denorm.cpu().numpy()
            action_sequences_np = action_sequences_np[..., :2]  # Only x,y for visualization
            action_label_np = action_label.cpu().numpy() if action_label is not None else None
            action_label_np = action_label_np[..., :2] if action_label_np is not None else None
            goal_pos_np = goal_pos.cpu().numpy() if goal_pos is not None else None
            
            # Visualize all samples before bucketing if bucketing was used
            if 'all_action_sequences' in results:
                all_action_sequences_np = get_action(results['all_action_sequences']).cpu().numpy()
                scores = [0.0]*all_action_sequences_np.shape[0]  # Placeholder scores for visualization
                self._visualize_action_sequences(
                    action_sequences=all_action_sequences_np,
                    curr_image=obs_images[-1],
                    batch_action_label=action_label_np,
                    batch_goal_pos=goal_pos_np,
                    dissimilarity_scores=scores,
                    save_dir=f"{save_dir}/all_samples",
                    viz_type=["image"],
                    **kwargs,
                )
            
            # Prepare data for visualization
            dissimilarity_scores_values = dissimilarity_scores.tolist()
            # if action_label_np is not None and goal_pos_np is not None:
            self._visualize_action_sequences(
                action_sequences=action_sequences_np,
                curr_image=obs_images[-1],
                batch_action_label=action_label_np,
                batch_goal_pos=goal_pos_np,
                dissimilarity_scores=dissimilarity_scores_values,
                save_dir=save_dir,
                viz_type=["graph","image"],
                **kwargs,
            )

            # future_images = []
            # for i in range(future_latents.shape[0]):
            #     state = self._vae_decode(future_latents[i])  # (num_steps, C, H, W)
            #     state = self._to_display_images(state, resize_hw=(240, 320))
            #     future_images.append(state)
            # future_images = np.stack(future_images)  # (num_samples, num_steps, C, H, W)

            # self._visualize_states(
            #     save_dir=save_dir,
            #     states=future_images,
            #     save_name="future_states.png",
            # )
            # self._visualize_states(
            #     save_dir=save_dir,
            #     states=obs_images,
            #     save_name="context_states.png",
            # )
        
        # Combine results with comprehensive state for next rollout
        combined_results = {
            # Action sequences and future predictions
            'action_sequences': best_action_sequence,
            'future_latents': future_latents,
            'rnd_loss': self.rnd_train_loss,
            # Dissimilarity and scoring
            'dissimilarity_score': best_score,
        }

        print("="*60)
        
        return combined_results
    
    def _visualize_all_samples_before_bucketing(
        self,
        action_sequences: np.ndarray,
        batch_action_label: np.ndarray,
        batch_goal_pos: np.ndarray,
        save_dir: str,
    ) -> None:
        """Visualize all action samples before bucketing with ground truth."""
        
        visualize_path = save_dir
        if not os.path.isdir(visualize_path):
            os.makedirs(visualize_path)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all sampled trajectories
        for i, traj in enumerate(action_sequences):
            color = plt.cm.rainbow(i / len(action_sequences))
            ax.plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.4, color=color, linewidth=1)
        
        # Plot ground truth trajectory (highlighted)
        ax.plot(batch_action_label[:, 0], batch_action_label[:, 1], 'o-', 
                color='magenta', linewidth=3, alpha=0.8, label='Ground Truth')
        
        # Plot start and goal
        ax.plot(0, 0, 'g*', markersize=15, label='Start')
        ax.plot(batch_goal_pos[0], batch_goal_pos[1], 'b*', markersize=15, label='Goal')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'All {len(action_sequences)} Samples Before Bucketing', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Save figure
        save_path = os.path.join(visualize_path, f"all_samples_before_bucketing.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _visualize_action_sequences(
        self,
        curr_image: np.ndarray,
        action_sequences: np.ndarray,
        batch_action_label: Optional[np.ndarray],
        batch_goal_pos: Optional[np.ndarray],
        dissimilarity_scores: List[float],
        save_dir: str,
        viz_type: str = ["image"],  # or "graph"
        **kwargs,
    ) -> None:
        """Visualize action sequences colored by dissimilarity score."""
        
        visualize_path = save_dir
        if not os.path.isdir(visualize_path):
            os.makedirs(visualize_path)
            
        # Normalize dissimilarity scores to [0, 1] for colormap
        num_action_seqs = len(action_sequences)
        scores_array = np.array(dissimilarity_scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        score_range = max_score - min_score if max_score > min_score else 1.0
        normalized_scores = (scores_array - min_score) / score_range
        
        # Map normalized scores to colors (using viridis colormap for better distinction)
        cmap = plt.cm.viridis
        traj_colors = [cmap(score) for score in normalized_scores] 
        traj_alphas = [1.0] * len(action_sequences)
        # Create labels with dissimilarity scores
        traj_labels = [f"Traj {i}: dissim={dissimilarity_scores[i]:.3f}" for i in range(num_action_seqs)]

        # Make points numpy array of robot positions and goal positions
        point_list = [np.array([0, 0])]
        point_colors = ["green"]
        point_alphas = [1.0]

        if batch_action_label is not None:
            traj_colors.append("magenta")
            traj_alphas.append(1.0)
            traj_labels.append("GT")
            point_list.append(batch_goal_pos)
            point_colors.append("blue")
            point_alphas.append(1.0)

            traj_list = np.concatenate([
                action_sequences,
                batch_action_label[None],
            ], axis=0)
        else:
            traj_list = action_sequences

        if "graph" in viz_type:
            # Create figure with trajectory plot and orientation arrows
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_trajs_and_points(
                ax,
                traj_list,
                point_list,
                traj_colors,
                point_colors,
                traj_labels=traj_labels,
                point_labels=["Start", "Goal"] if batch_goal_pos is not None else ["Start"],
                quiver_freq=0,
                traj_alphas=traj_alphas,
                point_alphas=point_alphas,
            )
            
            # Add legend
            ax.legend(loc='best', fontsize=9)
            
            # Add orientation arrows for each sample
            for sample_idx, traj in enumerate(action_sequences):
                # Calculate yaw angles for this trajectory
                angles = []
                for i in range(len(traj)):
                    dx, dy = traj[i]
                    angle = np.arctan2(dy, dx)
                    angles.append(angle)
                
                # Get color for this trajectory (matching the line color)
                arrow_color = traj_colors[sample_idx]
                
                # Draw arrows at each waypoint
                for i in range(len(traj)):
                    # Arrow length proportional to distance from origin
                    arrow_length = 0.15
                    angle = angles[i]
                    dx_arrow = arrow_length * np.cos(angle)
                    dy_arrow = arrow_length * np.sin(angle)
                    
                    ax.arrow(traj[i, 0], traj[i, 1], dx_arrow, dy_arrow,
                            head_width=0.08, head_length=0.06, fc=arrow_color, ec=arrow_color, alpha=0.6)
            
            ax.set_title(
                "Trajectories colored by Visual Dissimilarity Score\n(Higher score = More novel)",
                fontsize=14,
            )
            # Save figure
            save_path = os.path.join(visualize_path, f"sample_actions.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        if "image" in viz_type:
            fig, ax = plt.subplots(figsize=(8, 6))
            curr_image = np.moveaxis(curr_image, 0, -1) # (H, W, C)
            # Unnormalize if needed
            traj_list = traj_list * kwargs["metric_waypoint_spacing"] * self.config["waypoint_spacing"]

            # Plot both unconditioned and goal-conditioned actions on obs image
            plot_trajs_and_points_on_image(
                ax,
                curr_image, 
                self.dataset_name,
                traj_list,
                point_list,
                traj_colors=traj_colors,
                point_colors=point_colors,
            )
            ax.axis('off')

            save_path = os.path.join(visualize_path, f"samples_on_image.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
