import torch

def camera_position_from_spherical_angles(distance: float, elevation: float, azimuth: float, degrees: bool = True, device: str = "cpu"):
    """
    Generates the location of a camera in 3D space based on its distance, elevation, and azimuth angles.

    Args:
        distance (float): The distance from the target point.
        elevation (float): The elevation angle in either degrees or radians.
        azimuth (float): The azimuth angle in either degrees or radians.
        degrees (bool): Flag indicating whether the angles are in degrees (True) or radians (False).
        device (str): The device on which to perform the computations, either "cpu" or "cuda".

    Returns:
        torch.Tensor: A tensor representing the (N, 3) xyz location of the camera.
    """
    # Convert angles to radians if not in radians
    elevation_radians = (elevation * (math.pi / 180.0)) if degrees else elevation
    azimuth_radians = (azimuth * (math.pi / 180.0)) if degrees else azimuth

    # Calculate the spherical coordinates
    distance_vector = torch.tensor([distance, 0, 0], dtype=torch.float32, device=device)
    elevation_vector = torch.tensor([0, distance * torch.sin(elevation_radians), 0], dtype=torch.float32, device=device)
    azimuth_vector = torch.tensor([distance * torch.cos(elevation_radians) * torch.sin(azimuth_radians),
                                  distance * torch.cos(elevation_radians) * torch.cos(azimuth_radians),
                                  distance * torch.sin(elevation_radians)],
                                 dtype=torch.float32, device=device)

    # Combine the vectors to get the camera position
    camera_position = torch.stack((azimuth_vector, elevation_vector, distance_vector), dim=1)

    return camera_position

if __name__ == "__main__":
    import math
    import numpy as np

    # Example usage
    distance = 10.0
    elevation = 30.0  # in degrees
    azimuth = 45.0   # in degrees
    degrees = True
    device = "cpu"

    # Convert angles to radians for consistency
    elevation_radians = math.radians(elevation)
    azimuth_radians = math.radians(azimuth)

    # Create a tensor with shape (N, 1) for broadcasting
    N = 1  # Modify this value to simulate multiple inputs
    input_values = torch.tensor([[distance], [elevation_radians], [azimuth_radians]], dtype=torch.float32, device=device)

    # Call the function
    camera_position = camera_position_from_spherical_angles(**vars(input_values.squeeze()), degrees=degrees, device=device)

    # Print the result
    print(f"Camera Position: {camera_position}")