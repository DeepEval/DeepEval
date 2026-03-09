import torch

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
  V = torch.zeros_like(log_rotation)
  V[..., 0, 0] = 1 + torch.sin(rotation_angles) * log_rotation_hat[..., 0, 0]  + (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 0, 0]
  V[..., 0, 1] = torch.sin(rotation_angles) * log_rotation_hat[..., 0, 1] - (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 0, 1]
  V[..., 1, 0] = torch.sin(rotation_angles) * log_rotation_hat[..., 1, 0] + (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 1, 0]
  V[..., 1, 1] = 1 + torch.sin(rotation_angles) * log_rotation_hat[..., 1, 1] + (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 1, 1]
  V[..., 2, 2] = 1 + torch.sin(rotation_angles) * log_rotation_hat[..., 2, 2] + (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 2, 2]
  V[..., 2, 3] = torch.sin(rotation_angles) * log_rotation_hat[..., 2, 3] - (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 2, 3]
  V[..., 3, 2] = torch.sin(rotation_angles) * log_rotation_hat[..., 3, 2] + (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 3, 2]
  V[..., 3, 3] = 1 + torch.sin(rotation_angles) * log_rotation_hat[..., 3, 3] + (1 - torch.cos(rotation_angles)) * log_rotation_hat_square[..., 3, 3]
  
  return V 

if __name__ == "__main__":
    log_rotation = torch.randn(3, 3)
    log_rotation_hat = torch.randn(3, 3)
    log_rotation_hat_square = torch.randn(3, 3)
    rotation_angles = torch.rand(3) * 2 * 3.14159
    V = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles)
    print(V)