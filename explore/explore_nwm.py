#!/usr/bin/env python3
"""
Exploration server with streaming data via ZMQ.

This script provides an exploration server that:
1. Continuously listens for images via ZMQ
2. On first message: initializes exploration state
3. On subsequent messages: updates state, runs rollout, sends back denormalized actions
4. Works with an external finite state machine that controls when images are sent

The server is reactive - it processes images when received and sends actions back.
"""

from vanderer_paths import PATHS
import argparse
import os
import sys
import signal
import time
import json
import h5py
import torch
import torchvision.transforms.functional as TF
import numpy as np
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image
# ZMQ imports
import zmq
import msgpack

# Add paths

from noise_guide_naviexplorer import PriorNaviExplorer as GuideNaviExplorer, NoiseConfig
# from step_naviexplorer import NaviExplorer
from step_nwm import NaviExplorer


@dataclass
class ExplorationConfig:
    """Configuration for exploration server."""
    rollout_steps: int = 8             # Steps per single rollout
    context_size: int = 4              # Number of context images to maintain
    mode: str = "guide"                # "guide", "select", or "random"
    visualize: bool = False
    save_dir: str = "exploration_output"
    mask_type: str = "goal_mask"
    zmq_address: str = "tcp://127.0.0.1:5555"          # Address for receiving observations
    action_send_address: str = "tcp://127.0.0.1:5556"  # Address for sending actions
    # Image dimensions
    image_height: int = 224             # Image height for reconstruction
    image_width: int = 224              # Image width for reconstruction


class ZMQComm:
    """
    Simple ZMQ communication for receiving images and sending actions.
    Uses blocking recv() - no background threads needed.
    """
    def __init__(
        self, 
        image_address: str,
        action_address: str,
        image_height: int = 224,
        image_width: int = 224,
    ):
        self.image_height = image_height
        self.image_width = image_width
        
        # Initialize ZMQ context
        self.ctx = zmq.Context()
        
        # Socket for receiving images (PULL for point-to-point)
        self.image_sock = self.ctx.socket(zmq.PULL)
        self.image_sock.bind(image_address)
        print(f"[ZMQComm] Image receiver bound to {image_address}")
        
        # Socket for sending actions (PUSH for point-to-point)
        self.action_sock = self.ctx.socket(zmq.PUSH)
        self.action_sock.bind(action_address)
        print(f"[ZMQComm] Action sender bound to {action_address}")
    
    def receive_images(self) -> Tuple[List[Dict], bool]:
        """
        Receive images from ZMQ. Blocks until data arrives.
        
        Returns:
            List of observation dictionaries sorted by timestamp (oldest first)
        """
        # Receive multipart message: timestamps JSON + image bytes
        metadata = self.image_sock.recv_json()
        timestamps = metadata['timestamps']
        lhc_flag = metadata['lhc_flag']
        image_bytes = self.image_sock.recv()
        
        print(f"[ZMQComm] Received {len(timestamps)} images")
        
        # Reconstruct stacked images (N, H, W, 3)
        N = len(timestamps)
        stacked_images = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
            N, self.image_height, self.image_width, 3
        )
        
        # Create observations with timestamps
        observations = []
        for i, ts in enumerate(timestamps):
            observations.append({
                'image': stacked_images[i],
                'timestamp': ts
            })
        
        # Sort by timestamp - OLDEST FIRST
        observations.sort(key=lambda x: x['timestamp'], reverse=False)
        
        return observations, lhc_flag
    
    def send_actions(self, action_sequence: np.ndarray) -> bool:
        """
        Send action sequence back via ZMQ.
        
        Args:
            action_sequence: Action sequence array of shape (len_traj, action_dim)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            message_data = {
                'actions': action_sequence.tolist(),
                'timestamp': time.time(),
            }
            msg = msgpack.packb(message_data, use_bin_type=True)
            self.action_sock.send(msg)
            print(f"[ZMQComm] Sent actions with shape {action_sequence.shape}")
            return True
        except Exception as e:
            print(f"[ZMQComm] Error sending actions: {e}")
            return False
    
    def close(self):
        """Close ZMQ sockets and context."""
        self.image_sock.close()
        self.action_sock.close()
        self.ctx.term()
        print("[ZMQComm] Closed")


def denormalize_actions(
    actions: np.ndarray,
    metric_waypoint_spacing: float,
    waypoint_spacing: int = 1
) -> np.ndarray:
    """
    Denormalize actions by multiplying with metric_waypoint_spacing * waypoint_spacing.
    
    During training, actions are normalized by dividing:
        actions[:, :2] /= metric_waypoint_spacing * waypoint_spacing
    
    So to denormalize, we multiply back:
        actions[:, :2] *= metric_waypoint_spacing * waypoint_spacing
    
    Args:
        actions: Action sequence of shape (len_traj, action_dim)
        metric_waypoint_spacing: Metric spacing from data_config (meters)
        waypoint_spacing: Waypoint spacing used during training (frames)
        
    Returns:
        Denormalized actions
    """
    denorm_actions = actions.copy()
    denorm_factor = metric_waypoint_spacing * waypoint_spacing
    denorm_actions[:, :2] *= denorm_factor
    
    print(f"[Denorm] Denormalized actions by factor {denorm_factor:.4f}")
    return denorm_actions


def preprocess_observation(
    observations: List[Dict],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert observations to context images and latents.
    
    Args:
        observations: List of observation dictionaries (should be sorted oldest first)
        generator: NaviExplorer instance
        device: Torch device
        
    Returns:
        Tuple of (obs_images, obs_latents)
        - obs_images: (C*context_size, H, W) in [-1, 1] (concatenated along channel dim)
        - obs_latents: (context_size, D, H, W)
    """
    
    # Preprocess all images
    images = []
    for obs in observations:
        assert obs['image'].shape[:2] == (224, 224), f"Expected image shape (224, 224, 3) but got {obs['image'].shape}"
        # Convert to tensor (C, H, W)
        img = TF.to_tensor(obs['image']).float()  # (3, H, W) in [0, 1]
        images.append(img)
    
    # Stack images for encoding: (context_size, 3, H, W)
    obs_images = torch.stack(images, dim=0)
    
    return obs_images.to(device)


class ExplorationServer:
    """
    Exploration server that listens for images and responds with actions.
    
    Works with an external FSM:
    - First message: initializes exploration state
    - Subsequent messages: updates state, runs rollout, sends denormalized actions
    """
    
    def __init__(
        self,
        generator: Any,
        config: ExplorationConfig,
        metric_waypoint_spacing: float,
    ):
        """
        Initialize the exploration server.
        
        Args:
            generator: NaviExplorer instance
            config: Exploration configuration
            metric_waypoint_spacing: Metric waypoint spacing for denormalization
        """
        self.generator = generator
        self.config = config
        self.device = generator.device
        self.viz_freq = 50  # Visualize every N steps
        
        # Load metric waypoint spacing for denormalization
        self.metric_waypoint_spacing = metric_waypoint_spacing
        
        # State tracking
        self.state = None
        self.is_initialized = False
        self.step_count = 0
        self._last_results = None
        
        # Initialize ZMQ communication
        self.comm = ZMQComm(
            image_address=config.zmq_address,
            action_address=config.action_send_address,
            image_height=config.image_height,
            image_width=config.image_width,
        )
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        self.noise_stats = {}
        self.lhc_h5_path = os.path.join(self.config.save_dir, "long_horizon_embeddings.h5")
        self._num_saved_lhc_embeddings = 0
        self._lhc_initial_capacity = 5000
        self._lhc_growth_factor = 2
        
        print(f"[ExplorationServer] Initialized")
        print(f"  - Receiving images from: {config.zmq_address}")
        print(f"  - Sending actions to: {config.action_send_address}")
        print(f"  - Long-horizon embedding log: {self.lhc_h5_path}")

    @property
    def is_guide_mode(self) -> bool:
        return self.config.mode == "guide"

    def _append_noise_stats(self, results: Dict) -> None:
        """Accumulate noise distribution statistics for the current step in memory."""
        noise_mu = results.get('noise_mu')
        noise_var = results.get('noise_var')
        if noise_mu is None or noise_var is None:
            return

        self.noise_stats[self.step_count] = {
            'noise_mu': noise_mu.detach().cpu().tolist(),
            'noise_var': noise_var.detach().cpu().tolist(),
        }

    def _append_lhc_embeddings_to_h5(self, embeddings: torch.Tensor) -> None:
        """Append long-horizon context embeddings to a preallocated HDF5 file."""
        if embeddings is None or embeddings.numel() == 0:
            return

        embeddings_np = embeddings.detach().cpu().float().numpy()
        if embeddings_np.ndim == 3:
            embeddings_np = np.expand_dims(embeddings_np, axis=0)
        steps_np = np.full((embeddings_np.shape[0],), self.step_count, dtype=np.int64)

        with h5py.File(self.lhc_h5_path, "a") as f:
            if "embeddings" not in f:
                capacity = max(self._lhc_initial_capacity, embeddings_np.shape[0])
                f.create_dataset(
                    "embeddings",
                    shape=(capacity,) + embeddings_np.shape[1:],
                    dtype=embeddings_np.dtype,
                    maxshape=(None,) + embeddings_np.shape[1:],
                    chunks=(1,) + embeddings_np.shape[1:],
                    compression="gzip",
                )
                f.create_dataset(
                    "step",
                    shape=(capacity,),
                    dtype=np.int64,
                    maxshape=(None,),
                    chunks=True,
                )

            ds = f["embeddings"]
            step_ds = f["step"]
            if ds.shape[1:] != embeddings_np.shape[1:]:
                raise ValueError(
                    "Embedding shape mismatch while appending to HDF5: "
                    f"existing={ds.shape[1:]}, new={embeddings_np.shape[1:]}"
                )

            write_start = self._num_saved_lhc_embeddings
            write_end = write_start + embeddings_np.shape[0]

            if write_end > ds.shape[0]:
                new_capacity = ds.shape[0]
                while new_capacity < write_end:
                    new_capacity *= self._lhc_growth_factor
                ds.resize((new_capacity,) + ds.shape[1:])
                step_ds.resize((new_capacity,))

            ds[write_start:write_end] = embeddings_np
            step_ds[write_start:write_end] = steps_np

        self._num_saved_lhc_embeddings += embeddings_np.shape[0]
        print(
            f"[ExplorationServer] Saved {embeddings_np.shape[0]} embedding(s) "
            f"to {self.lhc_h5_path} (total={self._num_saved_lhc_embeddings})"
        )
    
    def _initialize_state(self, observations: List[Dict]) -> Dict:
        """Initialize exploration state from first observations."""
        print(f"[ExplorationServer] Initializing state with {len(observations)} observations")
        
        if len(observations) != self.config.context_size:
            print(f"Warning: Expected {self.config.context_size} observations but got {len(observations)}")
        
        # Convert to context
        obs_images = preprocess_observation(observations, self.device)
        
        return {
            # Short context for local planning
            'obs_images': obs_images,
            
            # Long horizon context for dissimilarity scoring (start with last obs_latent)
            'long_horizon_context': None,

            # Visualization data
            'goal_image': None,
            'goal_yaw': None,
        }
    
    def _update_state(self, observations: List[Dict], results: Dict) -> Dict:
        """Update exploration state with new observations after rollout."""
        print(f"[ExplorationServer] Updating state with {len(observations)} observations")
        
        if len(observations) != self.config.context_size:
            print(f"Warning: Expected {self.config.context_size} observations but got {len(observations)}")
        
        # Convert streaming observations to context
        obs_images = preprocess_observation(observations, self.device)

        # Adjust context size to cap max size by subsampling
        long_horizon_context = results['long_horizon_context']

        next_state = {
            'obs_images': obs_images,
            'long_horizon_context': long_horizon_context,
            'goal_image': self.state['goal_image'],
            'goal_yaw': self.state['goal_yaw'],
        }

        if self.is_guide_mode:
            if results.get('noise_mu') is not None and results.get('noise_var') is not None:
                noise_mu = results['noise_mu']
                noise_var = results['noise_var']
                print(f"Noise stats - Mean of mu: {noise_mu.mean().item():.4f}, Trace of var: {noise_var.mean().item():.4f}")

        return next_state
    
    def _run_rollout(self, lhc_flag: bool) -> Dict:
        """Run a single autoregressive rollout with current state."""
        step_save_dir = os.path.join(self.config.save_dir, f"step_{self.step_count}")
        
        # # Adjust noise distribution for exploration
        # self.generator.adjust_noise_distribution()
        
        # Run autoregressive rollout
        rollout_kwargs = {
            'obs_images': self.state['obs_images'],
            'goal_image': None,
            'long_horizon_context': self.state['long_horizon_context'],
            'goal_yaw': None,
            'action_label': None,
            'goal_pos': None,
            'num_samples': 20,
            'num_outputs': 6,
            'rollout_steps': self.config.rollout_steps,
            'visualize': (self.step_count % self.viz_freq == 0),
            'save_dir': step_save_dir,
            'mask_type': self.config.mask_type,
            'metric_waypoint_spacing': self.metric_waypoint_spacing,
            'lhc_flag': lhc_flag,
        }

        if not self.is_guide_mode:
            rollout_kwargs.update({
                'strategy': self.config.mode,
            })
        opt_actions = self.generator.autoregressive_rollout(**rollout_kwargs)
        return opt_actions
    
    def process_observations(self, observations: List[Dict], lhc_flag: bool) -> np.ndarray:
        """
        Process incoming observations and return actions.
        
        This is the main entry point for processing a single step:
        1. If first call: initialize state
        2. Run rollout to get actions
        3. Update state for next iteration
        4. Return actions
        
        Args:
            observations: List of observation dictionaries (sorted oldest first)
            lhc_flag: Flag indicating whether to add frame to long horizon context
        Returns:
            Action sequence of shape (len_traj, action_dim)
        """
        self.step_count += 1
        step_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"[ExplorationServer] Processing step {self.step_count}")
        print(f"{'='*60}")
        
        if not self.is_initialized:
            # First message: initialize state
            self.state = self._initialize_state(observations)
            self.is_initialized = True
            print(f"[ExplorationServer] State initialized")
        else:
            self.state = self._update_state(observations, self._last_results)
        
        if self.state['long_horizon_context'] is not None:
            print(f"Long horizon context size: {self.state['long_horizon_context'].shape[0]}")
        
        # Run rollout
        results = self._run_rollout(lhc_flag)
        self._last_results = results  # Store for next update
        self._append_noise_stats(results)

        # Persist latest context embedding when lhc_flag is enabled.
        updated_long_horizon_context = results.get('long_horizon_context')
        if (
            lhc_flag
            and updated_long_horizon_context is not None
        ):
            latest_embedding = updated_long_horizon_context[-1, ...]
            self._append_lhc_embeddings_to_h5(latest_embedding)
        
        print(f"Best trajectory selected with score: {results['dissimilarity_score']:.4f}")
        
        # Get actions and denormalize
        actions = results['action_sequences'].cpu().numpy()
        
        # Denormalize actions using metric waypoint spacing
        denorm_actions = denormalize_actions(
            actions, 
            self.metric_waypoint_spacing,
            waypoint_spacing=self.generator.config["waypoint_spacing"]
        )
        
        step_time = time.time() - step_start_time
        print(f"[ExplorationServer] Step {self.step_count} completed in {step_time:.2f}s")
        print(f"[ExplorationServer] Returning denormalized actions of shape {denorm_actions.shape}")
        
        return denorm_actions
    
    def run(self):
        """
        Main server loop: wait for images, process them, send actions.
        Uses blocking recv() - runs until interrupted.
        """
        print(f"\n[ExplorationServer] Starting main loop...")
        print(f"[ExplorationServer] Waiting for images (blocking)...")
        
        try:
            while True:
                # Block until images arrive
                observations, lhc_flag = self.comm.receive_images()
                
                # Process observations and get actions
                actions = self.process_observations(observations, lhc_flag)
                print(f"Actions shape: {actions.shape}")
                # Send actions back
                self.comm.send_actions(actions)

                # Sleep to avoid busy loop (optional)
                time.sleep(0.1)
                
        except (KeyboardInterrupt, SystemExit, Exception) as e:
            print(f"\n[ExplorationServer] Terminated: {e}")
            if self.noise_stats:
                save_path = os.path.join(self.config.save_dir, "noise_stats.json")
                with open(save_path, 'w') as f:
                    json.dump(self.noise_stats, f)
                print(f"[ExplorationServer] Saved noise stats to {save_path}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of the server."""
        print(f"[ExplorationServer] Shutting down...")
        self.comm.close()
        if self._num_saved_lhc_embeddings > 0:
            with h5py.File(self.lhc_h5_path, "a") as f:
                f["embeddings"].resize((self._num_saved_lhc_embeddings,) + f["embeddings"].shape[1:])
                f["step"].resize((self._num_saved_lhc_embeddings,))
            print(
                f"[ExplorationServer] Long-horizon embeddings saved: "
                f"{self._num_saved_lhc_embeddings} at {self.lhc_h5_path}"
            )
        
        # Save visualization if we have trajectory history
        if self.config.visualize and self.state and self.state.get('trajectory_history'):
            visualize_exploration(
                self.state['trajectory_history'],
                os.path.join(self.config.save_dir, "complete_exploration.png")
            )
        
        print(f"[ExplorationServer] Shutdown complete")
    
    def reset(self):
        """Reset the server state for a new exploration session."""
        print(f"[ExplorationServer] Resetting state...")
        self.state = None
        self.is_initialized = False
        self.step_count = 0
        self._last_results = None
        print(f"[ExplorationServer] State reset complete")


def visualize_exploration(
    exploration_history: List[Dict],
    save_path: str,
):
    """
    Visualize the complete exploration trajectory.
    
    Args:
        exploration_history: List of trajectory records
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Plot all trajectories
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(exploration_history)))
    
    cumulative_position = np.array([0.0, 0.0])
    all_positions = [cumulative_position.copy()]
    
    for i, record in enumerate(exploration_history):
        actions = record['actions']
        # Actions are cumulative positions from get_action, offset from current position
        trajectory = actions + cumulative_position
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'Step {i+1} (score: {record["dissimilarity_score"]:.3f})')
        
        # Update cumulative position for next trajectory
        cumulative_position = trajectory[-1]
        all_positions.append(cumulative_position.copy())
    
    all_positions = np.array(all_positions)
    ax.plot(0, 0, 'g*', markersize=15, label='Start')
    ax.plot(all_positions[-1, 0], all_positions[-1, 1], 'r*', markersize=15, label='End')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Complete Exploration Trajectory', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Right: Plot dissimilarity scores over time
    ax2 = axes[1]
    scores = [record['dissimilarity_score'] for record in exploration_history]
    steps = range(1, len(scores) + 1)
    
    ax2.plot(steps, scores, 'b-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Exploration Step', fontsize=12)
    ax2.set_ylabel('Dissimilarity Score', fontsize=12)
    ax2.set_title('Novelty Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Exploration visualization saved to {save_path}")


def main():
    def signal_handler(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    """Main entry point - creates and runs the exploration server."""

    parser = argparse.ArgumentParser(description="Exploration server with guide, select, and random modes.")
    parser.add_argument(
        "--town",
        type=str,
        required=True,
        help="Town to explore"
    )
    parser.add_argument(
        "--mode",
        choices=["guide", "select", "random"],
        default="guide",
        help="Explorer mode to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=f"{PATHS.repo}/outputs/ablations/wo_policy",
        help="Directory to save exploration outputs",
    )
    parser.add_argument(
        "--img-addr",
        type=str,
        default="tcp://127.0.0.1:5005",
        help="ZMQ address to receive images (e.g. tcp://127.0.1:5005)",
    )
    parser.add_argument(
        "--wp-addr",
        type=str,
        default="tcp://127.0.0.1:5006",
        help="ZMQ address to send waypoints (e.g. tcp://127.0.1:5006)",
    )
    args = parser.parse_args()
    args.save_dir = os.path.join(args.root_dir, f"{args.mode}", f"recorded_trajs_town_{args.town}_seed_{args.seed}")
    # Configuration
    model_config = f'{PATHS.repo}/train/config/naviexp_vae.yaml'
    data_config_path = f'{PATHS.repo}/train/vint_train/data/data_config.yaml'
    wm_checkpoint = f'{PATHS.nwm_logs}/nwm_cdit_b_finetune/checkpoints/latest.pth.tar'
    dataset_name = f"carla_town{args.town}"

    os.makedirs(args.save_dir, exist_ok=True)
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    metric_waypoint_spacing = data_config[dataset_name]["metric_waypoint_spacing"]
    
    # Server configuration
    config = ExplorationConfig(
        rollout_steps=8,
        context_size=4,
        mode=args.mode,
        visualize=False,
        save_dir=args.save_dir,
        mask_type="goal_mask",
        zmq_address=args.img_addr,
        action_send_address=args.wp_addr,
        image_height=224,
        image_width=224,
    )
    
    print("Loading NaviExplorer...")
    if args.mode != "select":
        raise NotImplementedError("Only 'select' mode is implemented in this script.")
    
    generator = NaviExplorer(
        config_path=model_config,
        wm_checkpoint=wm_checkpoint,
        device='cuda',
        dataset_name=dataset_name,
        waypoint_spacing=2,
    )
    
    # Create and run server
    server = ExplorationServer(generator=generator,
                               config=config,
                               metric_waypoint_spacing=metric_waypoint_spacing)
    server.run()


if __name__ == "__main__":
    main()
