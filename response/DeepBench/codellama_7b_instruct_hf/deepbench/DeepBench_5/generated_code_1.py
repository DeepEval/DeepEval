import torch
import math

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    # Compute the skew-symmetric matrix from the logarithm of the rotation matrix
    log_rotation_hat = log_rotation_hat.unsqueeze(1)
    log_rotation_hat_square = log_rotation_hat_square.unsqueeze(1)

    # Compute the sine and cosine of the angles of rotation
    sine_angles = torch.sin(rotation_angles)
    cosine_angles = torch.cos(rotation_angles)

    # Compute the first and second terms of the V matrix
    first_term = torch.exp(-log_rotation_hat_square / 2)
    second_term = torch.sin(rotation_angles) - log_rotation_hat * sine_angles

    # Compute the final V matrix
    V = first_term * (1 - cosine_angles) + second_term * sine_angles

    # Handle numerical stability
    V = V + eps * torch.eye(V.size(0))

    return V

if __name__ == "__main__":
    # Create sample input values
    log_rotation = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    log_rotation_hat = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    log_rotation_hat_square = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    rotation_angles = torch.tensor([0.1, 0.2, 0.3])

    # Call the function and print the results
    V = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles)
    print(V)