"""
Policy-only NoMaD rollout.

This module keeps the same external interface as explore/step_naviexplorer.py
for compatibility with explore_carla.py, but uses only the policy model.
During rollout it generates action candidates and returns one random trajectory.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.training.train_utils import get_action
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.visualizing.action_utils import plot_trajs_and_points, plot_trajs_and_points_on_image

class NaviExplorer:
    """Policy-only action generator with explore_carla-compatible API."""

    def __init__(
        self,
        dp_checkpoint: str,
        wm_checkpoint: str,
        config_path: str,
        dataset_name: str = None,
        waypoint_spacing: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.config = self._load_config(config_path)
        self.config["waypoint_spacing"] = waypoint_spacing

        self.wm_checkpoint = wm_checkpoint
        self.dataset_name = dataset_name

        self.model = self._build_model()
        self._load_checkpoint(dp_checkpoint)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self._denorm = get_action
        print(f"Policy-only NoMaD loaded on {self.device}")

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _build_model(self) -> NoMaD:
        config = self.config

        if config["vision_encoder"] != "nomad_vint":
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],
            obs_encoder=config["obs_encoder"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            use_cache_encodings=False,
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        noise_pred_net = ConditionalUnet1D(
            input_dim=3 if config["learn_angle"] else 2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )

        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
        model.eval()
        model.to(self.device)
        return model

    def _load_checkpoint(self, checkpoint_path: str):
        print(f"Loading policy model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        first_key = list(checkpoint.keys())[0]
        if "_orig_mod.module." in first_key:
            checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
        elif "_orig_mod." in first_key:
            checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}

        self.model.load_state_dict(checkpoint, strict=True)

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

    def _sample_noise(
        self,
        num_samples: int,
        action_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.randn(
            (num_samples, self.config["len_traj_pred"], action_dim),
            device=self.device,
            dtype=dtype,
        )

    def _prepare_effnet_inputs(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
    ) -> tuple:
        """Match train.py preprocessing for the EfficientNet encoder.

        obs_img: (context_size+1, 3, H, W) in [0, 1], oldest first.
        goal_img: (N, 3, H, W) or (3, H, W) in [0, 1]; only the last image is used.
        Returns obs (1, 3*(context_size+1), h, w) and goal (1, 3, h, w), resized to
        config image_size and ImageNet-normalized.
        """
        expected = self.config["context_size"] + 1
        assert obs_img.shape[0] == expected, f"Expected {expected} context images, got {obs_img.shape[0]}"
        if goal_img.dim() == 3:
            goal_img = goal_img.unsqueeze(0)
        goal_img = goal_img[-1:]

        width, height = self.config["image_size"]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        def prep(x: torch.Tensor) -> torch.Tensor:
            x = TF.resize(x, [height, width], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)  # PIL resize default in training
            return TF.normalize(x, mean=mean, std=std)

        obs_img = prep(obs_img).reshape(1, -1, height, width)
        goal_img = prep(goal_img)
        return obs_img, goal_img

    def _encode_condition(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        mask_type: str,
    ) -> torch.Tensor:
        if self.config["obs_encoder"].startswith("efficientnet"):
            obs_img, goal_img = self._prepare_effnet_inputs(obs_img, goal_img)
        batch_size = obs_img.shape[0]
        assert mask_type == "goal_mask", f"Only 'goal_mask' is supported in this implementation, got {mask_type}"
        goal_mask_tensor = torch.ones((batch_size,), dtype=torch.long, device=self.device)

        with torch.no_grad():
            cond = self.model(
                "vision_encoder",
                obs_img=obs_img,
                goal_img=goal_img,
                input_goal_mask=goal_mask_tensor,
            )
        return cond.flatten(start_dim=1)

    def generate_action_sequences(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        num_samples: int = 20,
        mask_type: str = "goal_mask",
        action_dim: int = 2,
    ) -> Dict:
        obs_cond = self._encode_condition(obs_img=obs_img, goal_img=goal_img, mask_type=mask_type)

        with torch.no_grad():
            obs_cond = obs_cond.repeat(num_samples, 1)
            self.noise_scheduler.set_timesteps(self.config["num_diffusion_iters"])

        naction = self._sample_noise(
            num_samples=num_samples,
            action_dim=action_dim,
            dtype=torch.float32,
        )

        for k in self.noise_scheduler.timesteps[:]:
            with torch.no_grad():
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                ).prev_sample

        return {"action_sequences": naction}

    def autoregressive_rollout(
        self,
        obs_images: torch.Tensor,
        goal_image: torch.Tensor = None,
        num_samples: int = 20,
        save_dir: str = "rollout_output",
        mask_type: str = "goal_mask",
        **kwargs,
    ) -> Dict:
        """
        Compatible with step_naviexplorer.autoregressive_rollout, but policy-only.

        Long-horizon context is intentionally kept unchanged and unused.
        """
        obs_images = obs_images.to(self.device)
        goal_image = goal_image.to(self.device) if goal_image is not None else obs_images[0]
        results = self.generate_action_sequences(
            obs_img=obs_images,
            goal_img=goal_image,
            num_samples=num_samples,
            mask_type=mask_type,
            action_dim=3 if self.config["learn_angle"] else 2,
        )
        action_sequences = results["action_sequences"]

        chosen_idx = np.random.randint(0, action_sequences.shape[0])
        action_sequences_denorm = get_action(action_sequences)
        chosen_action = action_sequences_denorm[chosen_idx]

        if kwargs.pop("visualize", True):
            action_sequences_np = action_sequences_denorm.cpu().numpy()
            action_sequences_np = action_sequences_np[..., :2]  # Only x,y for visualization
            self._visualize_action_sequences(
                    action_sequences=action_sequences_np,
                    curr_image=obs_images[-1].cpu().numpy(),
                    batch_action_label=None,
                    batch_goal_pos=None,
                    dissimilarity_scores=[0.0] * action_sequences_np.shape[0],
                    save_dir=save_dir,
                    viz_type=["graph","image"],
                    **kwargs,
                )

        return {
            "action_sequences": chosen_action,
        }
    

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
            
            print(f"Visualization saved to {visualize_path}")

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
