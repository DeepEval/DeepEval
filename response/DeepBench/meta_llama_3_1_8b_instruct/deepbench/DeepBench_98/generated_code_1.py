import numpy as np
from scipy.linalg import svd
import torch

def run_8point(points1, points2, weights):
    B, N, _ = points1.shape

    # Concatenate the points to form the A matrix
    A = np.zeros((N * B, 9))
    for i in range(B):
        for j in range(N):
            x1, y1 = points1[i, j]
            x2, y2 = points2[i, j]
            A[j * B + i, 0] = -x1 * x2
            A[j * B + i, 1] = -y1 * x2
            A[j * B + i, 2] = -x1 * y2
            A[j * B + i, 3] = -y1 * y2
            A[j * B + i, 4] = -x1 * x1
            A[j * B + i, 5] = -x1 * y1
            A[j * B + i, 6] = -y1 * x1
            A[j * B + i, 7] = -y1 * y1
            A[j * B + i, 8] = 1.0

    # Compute the weighted least squares solution
    W = np.diag(weights.reshape(-1))
    U, s, Vh = svd(np.dot(W, A.T))
    S = np.diag(1 / s)
    F = np.dot(Vh.T, np.dot(S, U.T)).T
    F = F.reshape(B, 3, 3)

    return torch.tensor(F)

if __name__ == "__main__":
    # Create sample input values
    points1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    points2 = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    weights = torch.tensor([[1, 1], [1, 1]])

    # Call the function
    F = run_8point(points1, points2, weights)

    # Print the results
    print(F)