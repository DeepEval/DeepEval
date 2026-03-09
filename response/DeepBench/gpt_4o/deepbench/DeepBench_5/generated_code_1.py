import torch

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    angle_squares = rotation_angles ** 2

    # Use Taylor expansion approximations for small angles
    A = torch.where(
        rotation_angles > eps,
        torch.sin(rotation_angles) / rotation_angles,
        1.0 - angle_squares / 6 + angle_squares**2 / 120
    )
    
    B = torch.where(
        rotation_angles > eps,
        (1 - torch.cos(rotation_angles)) / angle_squares,
        0.5 - angle_squares / 24 + angle_squares**2 / 720
    )
    
    C = torch.where(
        rotation_angles > eps,
        (1 - A) / angle_squares,
        1.0 / 6 - angle_squares / 120 + angle_squares**2 / 5040
    )
    
    V = torch.eye(3) + B.unsqueeze(-1) * log_rotation_hat + C.unsqueeze(-1) * log_rotation_hat_square
    return V

if __name__ == "__main__":
    log_rotation = torch.tensor([0.1, 0.2, 0.3])
    log_rotation_hat = torch.tensor([
        [0, -0.3, 0.2],
        [0.3, 0, -0.1],
        [-0.2, 0.1, 0]
    ])
    log_rotation_hat_square = torch.matmul(log_rotation_hat, log_rotation_hat)
    rotation_angles = torch.norm(log_rotation)

    V = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles)
    print("Computed V matrix:\n", V)