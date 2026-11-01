"""
ZMQ Communication class for CARLA waypoint controller.

Handles:
- Receiving waypoints from tcp://127.0.0.1:5556
- Collecting images with timestamps during waypoint execution
- Sending sampled images after all waypoints are reached
"""

import json
from typing import List, Tuple, Optional
from PIL import Image

import numpy as np
import zmq
import msgpack


class ZMQComm:
    """
    ZMQ Communication handler for CARLA.
    
    Workflow:
    1. Receive waypoints
    2. Collect all images with timestamps during execution
    3. When all waypoints reached, sample images at specified frequency
    4. Send sampled images (context_size number of frames)
    5. Wait for next waypoints
    """
    
    def __init__(
        self,
        waypoint_addr: str = "tcp://127.0.0.1:5556",
        image_addr: str = "tcp://127.0.0.1:5555",
        context_size: int = 4,
        context_spacing: float = 1.06,
    ):
        """
        Initialize ZMQ sockets and image buffer.
        
        Args:
            waypoint_addr: Address to receive waypoints from policy (PULL socket connects to policy's PUSH)
            image_addr: Address to send images to policy (PUSH socket connects to policy's PULL)
            sampling_freq_hz: Frequency (Hz) to sample images from buffer when sending
            context_size: Number of sampled images to send (context window)
        """
        self.waypoint_addr = waypoint_addr
        self.image_addr = image_addr
        self.context_spacing = context_spacing
        self.context_size = context_size

        # Image buffer: stores (rgb_image, location, timestamp) tuples
        self.image_buffer: List[Tuple[np.ndarray, Tuple[float, float, float], float]] = []
        
        # ZMQ context and sockets
        self.context = zmq.Context()
        
        # Socket to receive waypoints/actions from policy (PULL - connects to policy's PUSH)
        self.waypoint_socket = self.context.socket(zmq.PULL)
        self.waypoint_socket.connect(waypoint_addr)
        # No timeout - blocking receive
        
        # Socket to send images to policy (PUSH - connects to policy's PULL)
        self.image_socket = self.context.socket(zmq.PUSH)
        self.image_socket.connect(image_addr)
        
        print(f"[ZMQComm] Receiving waypoints/actions from {waypoint_addr}")
        print(f"[ZMQComm] Sending images to {image_addr}")
        print(f"[ZMQComm] Context: {context_size} frames")
    
    def add_image(self, rgb_image: np.ndarray,
                  location: Tuple[float, float, float],
                  timestamp: float) -> None:
        """
        Add an RGB image with timestamp to the buffer.
        Args:
            rgb_image: RGB image (H, W, 3)
            location: Location coordinates (x, y, z)
            timestamp: Timestamp for the image
        """
        # Make a copy to avoid reference issues
        self.image_buffer.append((rgb_image.copy(), location, timestamp))
    
    def clear_buffer(self) -> None:
        """Clear the image buffer."""
        self.image_buffer.clear()
        print(f"[ZMQComm] Buffer cleared")
    
    def sample_by_distance(self, buffer):
        """
        Sample images at self.context_spacing distance, anchored from the most recent timestamp.

        Walks backward from the latest location, and collects frames 
        at approx context_spacing distance intervals,

        Args:
            images: list of items (frames)
            locations: list/array of location coordinates (x, y, z)

        Returns:
            (sampled_images, sampled_locations) in chronological order (oldest first)
        """

        # Extract locations and images from buffer
        locations = [loc for _, loc, _ in buffer]
        locations = np.asarray(locations)  # Convert to numpy array for distance calculations

        if len(locations) == 0:
            return [], [], []

        start_location = locations[-1]
        sampled_locations = [start_location]
        sampled_indices = [len(buffer) - 1]
            
        for i in range(len(buffer)-2,-1,-1):
            dist = np.linalg.norm(locations[i] - sampled_locations[0])
            if dist >= self.context_spacing:
                sampled_locations.insert(0, locations[i])  # Insert at front to maintain order
                sampled_indices.insert(0, i)
                if len(sampled_indices) == self.context_size:
                    break
        
        return [self.image_buffer[i] for i in sampled_indices]
    
    def get_context_images(self) -> List[Tuple[float, np.ndarray]]:
        """
        Return the last context_size images from the buffer.

        Images are already distance-spaced by the caller, so no
        frequency-based resampling is needed.

        Returns:
            List of (timestamp, image) tuples in chronological order (oldest first),
            with at most context_size frames.
        """
        if not self.image_buffer:
            return [], [], []

        # Sample images by distance
        sampled_buffer = self.sample_by_distance(self.image_buffer)

        print(
            f"[ZMQComm] Buffer: {len(self.image_buffer)} frames, "
            f"returning last {len(sampled_buffer)} as context"
        )
        return sampled_buffer

    def send_images(self, long_horizon_flag: bool) -> bool:
        """
        Sample and send images from buffer via PUSH socket.
        
        Returns:
            True if images were sent, False if buffer is empty
        """
        sampled_buffer = self.get_context_images()

        if len(sampled_buffer) < self.context_size:
            print(f"[ZMQComm] Not enough context. Found {len(sampled_buffer)}, required {self.context_size}")
            # If buffer has fewer frames than context_size, return False
            return False
        
        # Extract timestamps and images
        # images = [img for _, img in samples]
        images = []
        for i in range(len(sampled_buffer)):
            img = Image.fromarray(sampled_buffer[i][0])  # Convert to PIL Image
            img = img.resize((224, 224))  # Resize to 224x224
            img = np.asarray(img)  # Convert back to numpy array
            images.append(img)
        
        # Stack images: shape (N, H, W, 3)
        stacked_images = np.stack(images, axis=0)
        
        # Send multipart: timestamps JSON + image bytes
        timestamps = [sampled_buffer[i][2] for i in range(len(sampled_buffer))]
        self.image_socket.send_json({"timestamps": timestamps, "lhc_flag": long_horizon_flag}, zmq.SNDMORE)
        self.image_socket.send(stacked_images.tobytes())
        
        print(f"[ZMQComm] Sent {len(sampled_buffer)} images with timestamps")
        return True
    
    def receive_waypoints(self) -> Optional[List[Tuple[float, float, float]]]:
        """
        Receive waypoints/actions from policy via PULL socket (BLOCKING).
        This will wait indefinitely until waypoints are received.
        
        Expected format (msgpack):
        {
            'actions': [[x, y, z], [x, y, z], ...],
            'timestamp': float
        }
        
        Returns:
            List of (x_fwd, y_left, z_up) tuples in FLU frame, or None on error
        """
        try:
            msg = self.waypoint_socket.recv()
            data = msgpack.unpackb(msg, raw=False)
            
            # Validate format
            if not isinstance(data, dict) or 'actions' not in data:
                print(f"[ZMQComm] Invalid waypoint format: expected dict with 'actions' key")
                return None
            
            waypoints = data['actions']
            timestamp = data.get('timestamp', None)
            
            if not isinstance(waypoints, list):
                print(f"[ZMQComm] Invalid waypoints format: expected list")
                return None
            
            # Convert to list of tuples
            result = []
            for wp in waypoints:
                if len(wp) == 2:
                    result.append((float(wp[0]), float(wp[1]), 0.0))  # Assume z=0 if not provided
                else:
                    result.append((float(wp[0]), float(wp[1]), float(wp[2])))
            if result:
                ts_str = f", timestamp={timestamp:.3f}" if timestamp else ""
                print(f"[ZMQComm] Received {len(result)} waypoints{ts_str}")
            return result if result else None
            
        except msgpack.exceptions.UnpackException as e:
            print(f"[ZMQComm] Failed to unpack msgpack: {e}")
            return None
        except Exception as e:
            print(f"[ZMQComm] Error receiving waypoints: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"[ZMQComm] Invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"[ZMQComm] Error receiving waypoints: {e}")
            return None
    
    def close(self) -> None:
        """Close ZMQ sockets and context."""
        self.waypoint_socket.close()
        self.image_socket.close()
        self.context.term()
        print("[ZMQComm] Sockets closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
