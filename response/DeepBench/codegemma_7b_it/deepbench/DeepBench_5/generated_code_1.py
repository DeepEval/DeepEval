import torch

def _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles, eps=1e-4):
    """
    Computes the "V" matrix used in SE(3) transformations.

    Args:
        log_rotation: The logarithm of the rotation matrix.
        log_rotation_hat: The skew-symmetric matrix derived from `log_rotation`.
        log_rotation_hat_square: The square of the skew-symmetric matrix.
        rotation_angles: The angles of rotation.
        eps: A small value for numerical stability, defaulting to 1e-4.

    Returns:
        A tensor `V` calculated using the provided inputs.
    """

    # Calculate the square of half the angles
    half_angles_square = 0.5 * rotation_angles ** 2

    # Calculate the V matrix using the provided formula
    V = (
        torch.eye(3)
        + (1 - torch.cos(rotation_angles)) / half_angles_square * log_rotation_hat
        + (
            (
                log_rotation_hat_square
                - (torch.sin(rotation_angles) / half_angles_square) * log_rotation_hat
            )
            / half_angles_square ** 2
        )
    )

    # Handle numerical stability
    V = V.clamp(min=eps)

    return V

if __name__ == "__main__":
    # Create sample input values
    log_rotation = torch.randn(3)
    log_rotation_hat = torch.tensor([[0, -log_rotation[2], log_rotation[1]],
                                      [log_rotation[2], 0, -log_rotation[0]],
                                      [-log_rotation[1], log_rotation[0], 0]])
    log_rotation_hat_square = log_rotation_hat.mm(log_rotation_hat)
    rotation_angles = torch.acos(0.5 * (log_rotation_hat_square.trace() - 1))

    # Call the function and print the results
    V = _se3_V_matrix(log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles)
    print(V)