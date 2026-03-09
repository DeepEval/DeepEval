import torch

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    theta = rotation_angles.unsqueeze(-1)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    V = torch.zeros((6, 6), device=log_rotation.device)

    V[0:3, 0:3] = log_rotation_hat
    V[0:3, 3:6] = torch.eye(3, device=log_rotation.device) * sin_theta / (theta + eps)
    V[3:6, 0:3] = log_rotation_hat_square * (1 - cos_theta) / (theta + eps).pow(2)
    V[3:6, 3:6] = torch.eye(3, device=log_rotation.device) * sin_theta

    return V

if __name__ == "__main__":
    log_rotation = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    log_rotation_hat = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    log_rotation_hat_square = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rotation_angles = torch.tensor([0.1])  # Example rotation angle

    V_matrix = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles)
    print(V_matrix)