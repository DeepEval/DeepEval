import torch
import numpy as np

def find_homography_lines_dlt(ls1, ls2, weights=None):
    """
    Computes the homography matrix using the DLT formulation for line correspondences.

    Args:
        ls1 (torch.Tensor): First set of line segments with shape (B, N, 2, 2).
        ls2 (torch.Tensor): Second set of line segments with shape (B, N, 2, 2).
        weights (torch.Tensor, optional): Optional tensor of weights per point correspondence with shape (B, N).

    Returns:
        homography_matrix (torch.Tensor): Computed homography matrix with shape (B, 3, 3).
    """
    # Compute the matrix of line-line correspondences
    A = torch.zeros((4, 9), dtype=torch.float32)
    for i in range(ls1.shape[0]):
        for j in range(ls1.shape[1]):
            p1 = ls1[i, j, 0]
            p2 = ls1[i, j, 1]
            p3 = ls2[i, j, 0]
            p4 = ls2[i, j, 1]
            A[i, :] = torch.cat((p1[0], p2[0], p3[0], p4[0], p1[1], p2[1], p3[1], p4[1], torch.ones(1)))

    # Compute the matrix of weights
    if weights is not None:
        W = torch.zeros((4, 4), dtype=torch.float32)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                W[i, j] = weights[i, j]
    else:
        W = torch.ones((4, 4), dtype=torch.float32)

    # Solve the linear system using the Weighted Least Squares Solution for the 4 Line correspondences algorithm
    homography_matrix = torch.linalg.solve(A, W)

    return homography_matrix

# Test the function
if __name__ == "__main__":
    # Create sample input values
    ls1 = torch.tensor([[[[0, 0], [1, 0]], [[1, 0], [2, 0]]]])
    ls2 = torch.tensor([[[[0, 1], [1, 1]], [[1, 1], [2, 1]]]])
    weights = torch.tensor([[[1, 2], [3, 4]]])

    # Compute the homography matrix
    homography_matrix = find_homography_lines_dlt(ls1, ls2, weights)

    # Print the results
    print(homography_matrix)