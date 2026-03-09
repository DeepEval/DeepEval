import torch
import torch.nn.functional as F
import math

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    # Compute sin(rotation_angles) and cos(rotation_angles) for the V matrix calculation
    sin_rotation_angles = torch.sin(rotation_angles)
    cos_rotation_angles = torch.cos(rotation_angles)

    # Compute the V matrix
    V = torch.cat([
        log_rotation,
        sin_rotation_angles * log_rotation_hat,
        (1 - cos_rotation_angles) * log_rotation_hat_square + (eps - sin_rotation_angles) * log_rotation_hat_square @ log_rotation_hat
    ], dim=-1)

    return V

if __name__ == "__main__":
    # Create sample input values
    log_rotation = torch.tensor([[0.1, 0.2, 0.3]]).float()
    log_rotation_hat = torch.tensor([[0, -0.3, 0.2], [0.3, 0, 0.1], [-0.2, -0.1, 0]]).float()
    log_rotation_hat_square = torch.tensor([[0, -0.06, 0.04], [0.06, 0, 0.02], [-0.04, -0.02, 0]]).float()
    rotation_angles = torch.tensor([0.5]).float()

    # Call the function and print the results
    V = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles)
    print(V)