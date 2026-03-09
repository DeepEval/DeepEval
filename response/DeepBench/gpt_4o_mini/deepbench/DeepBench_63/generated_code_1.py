import torch

def project_points_orthographic(points_in_camera):
    projected_points = points_in_camera[:, :2]  # Extract x and y coordinates
    return projected_points

if __name__ == "__main__":
    sample_points = torch.tensor([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0],
                                   [7.0, 8.0, 9.0]])
    projected_points = project_points_orthographic(sample_points)
    print(projected_points)