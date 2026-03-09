import torch

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    # Convert angles to radians
    rotation_angles = rotation_angles * (180 / torch.pi)
    
    # Compute the V matrix
    V = torch.zeros_like(log_rotation)
    for angle in rotation_angles:
        angle_rad = angle - eps  # to avoid sin(0) = 0 issue
        skew_symmetric = torch.stack([
            -angle_rad * torch.cos(log_rotation[:, 1]) * torch.eye(3),
            angle_rad * (torch.cos(log_rotation[:, 0]) * torch.eye(3) - torch.sin(log_rotation[:, 0]) * torch.eye(3)),
            angle_rad * (torch.sin(log_rotation[:, 0]) * torch.eye(3) + torch.cos(log_rotation[:, 0]) * torch.eye(3))
        ], dim=1)
        V += skew_symmetric
    
    V += torch.eye(3).unsqueeze(0).unsqueeze(0).to(log_rotation.device)
    
    return V

if __name__ == "__main__":
    # Sample input values
    batch_size = 4
    rotation_angles = torch.tensor([[0, 90, 45]], dtype=torch.float32).view(1, 3, 1)
    log_rotation = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    log_rotation_hat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)
    log_rotation_hat_square = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32)

    # Call the function and print the results
    V = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles)
    print(V)