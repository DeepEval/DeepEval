import torch
from torch.linalg import lstsq

def run_8point(points1: torch.Tensor, points2: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the fundamental matrix using the DLT formulation. The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`."
    """

    B, N = points1.shape[:2]

    # Calculate the weighted sum of cross products between the corresponding point pairs
    Xp = (points2[:, :, 0] - points1[:, :, 0]) * weights
    Yp = (points2[:, :, 1] - points1[:, :, 1]) * weights
    X = (points1[:, :, 0] * points2[:, :, 0] - points1[:, :, 1] * points2[:, :, 1]) * weights
    Y = (points1[:, :, 0] * points2[:, :, 1] + points1[:, :, 1] * points2[:, :, 0]) * weights

    # Construct the A matrix for the weighted least squares problem
    A = torch.stack((
        torch.cat([X, -Yp, points1[:, :, 0]], dim=1),
        torch.cat([Y, Xp, points1[:, :, 1]], dim=1),
        torch.cat([points1[:, :, 0] * X, points1[:, :, 1] * X, X], dim=1),
        torch.cat([points1[:, :, 0] * Y, points1[:, :, 1] * Y, Y], dim=1)
    ), dim=1)

    # Solve the weighted least squares problem
    u, _, vt = lstsq(A.transpose(1, 2), torch.zeros(B, 4).to(points1.device))

    # Reshape and return the fundamental matrix
    return u.view(B, 3, 3)

if __name__ == "__main__":
    # Create sample input values
    B = 2
    N = 10
    points1 = torch.randn(B, N, 2)
    points2 = torch.randn(B, N, 2)
    weights = torch.rand(B, N)

    # Call the function and print the results
    fundamental_matrix = run_8point(points1, points2, weights)
    print(fundamental_matrix)