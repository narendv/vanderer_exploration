import carla
import numpy as np

def serialize_transform(t: carla.Transform):
    """Serialize a transform to a 6-element tuple."""
    return (t.location.x, t.location.y, t.location.z, 
            t.rotation.pitch, t.rotation.yaw, t.rotation.roll)

def generate_spawn_points(town_name: str, host: str = 'localhost', port: int = 2000):
    """
    Connect to CARLA, load a town, and save its spawn points.
    
    Args:
        town_name: Name of the town (e.g., 'Town10HD', 'Town04')
        host: CARLA server host
        port: CARLA server port
    """
    # Connect to CARLA
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    
    # Load the world
    world = client.load_world(town_name)
    
    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    print(f"Found {len(spawn_points)} spawn points in {town_name}")
    
    # Serialize spawn points
    serialized_spawns = np.array([serialize_transform(sp) for sp in spawn_points])
    
    # Save to file
    output_file = f"{town_name}_spawns.npy"
    np.save(output_file, serialized_spawns)
    print(f"Saved spawn points to {output_file}")
    print(f"Shape: {serialized_spawns.shape}")
    
    return serialized_spawns

if __name__ == "__main__":
    # Make sure CARLA server is running before executing this
    env_list = ['Town01', 'Town07']
    for env in env_list:
        print(f"Generating spawn points for {env}...")
        generate_spawn_points(env)
    