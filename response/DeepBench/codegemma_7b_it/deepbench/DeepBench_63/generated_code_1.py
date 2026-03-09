import torch

def project_points_orthographic(points_in_camera):
    # Convert points to homogeneous coordinates
    points_in_camera_homog = torch.cat((points_in_camera, torch.ones_like(points_in_camera[:, :1])), dim=1)

    # Project to canonical z=1 plane
    projected_points_homog = points_in_camera_homog / points_in_camera_homog[:, 2:3]

    # Remove homogeneous coordinate
    projected_points = projected_points_homog[:, :2]

    return projected_points

if __name__ == "__main__":
    # Sample input points
    points_in_camera = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

    # Project points
    projected_points = project_points_orthographic(points_in_camera)

    # Print results
    print("Input points:")
    print(points_in_camera)

    print("Projected points:")
    print(projected_points)