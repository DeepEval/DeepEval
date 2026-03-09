import torch

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    V_1 = torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)
    V_1[0:3, 0:3] += log_rotation_hat_square / (1 + eps - log_rotation.pow(2).sum(dim=-1, keepdim=True))
    V_1[0:3, 0:3] += log_rotation[:, None, None] * log_rotation_hat / (1 + eps - log_rotation.pow(2).sum(dim=-1, keepdim=True))
    
    V_2 = torch.zeros((log_rotation.shape[0], 3, 3), dtype=log_rotation.dtype, device=log_rotation.device)
    V_2[:, 0, 1] = -log_rotation[:, 2]
    V_2[:, 0, 2] = log_rotation[:, 1]
    V_2[:, 1, 0] = log_rotation[:, 2]
    V_2[:, 1, 2] = -log_rotation[:, 0]
    V_2[:, 2, 0] = -log_rotation[:, 1]
    V_2[:, 2, 1] = log_rotation[:, 0]
    
    V = V_1 + V_2
    return V

if __name__ == "__main__":
    # Example usage
    log_rotation = torch.tensor([[0.1, 0.2, 0.3]])
    log_rotation_hat = torch.tensor([[
        [0, -0.3, 0.2],
        [0.3, 0, -0.1],
        [-0.2, 0.1, 0]
    ]])
    log_rotation_hat_square = torch.matmul(log_rotation_hat, log_rotation_hat)
    rotation_angles = log_rotation.norm(dim=-1)
    eps = 1e-4

    V = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps)
    print(V)