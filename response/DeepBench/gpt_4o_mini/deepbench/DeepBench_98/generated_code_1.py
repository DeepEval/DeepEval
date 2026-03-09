import numpy as np
import torch

def run_8point(points1, points2, weights):
    B, N, _ = points1.shape
    A = torch.zeros((B, N, 9), dtype=points1.dtype)

    for i in range(N):
        x1, y1 = points1[:, i, 0], points1[:, i, 1]
        x2, y2 = points2[:, i, 0], points2[:, i, 1]
        A[:, i, :] = weights[:, i].view(-1, 1) * torch.stack((x2 * x1, x2 * y1, x2, 
                                                                y2 * x1, y2 * y1, y2,
                                                                x1, y1, torch.ones_like(x1)), dim=1)

    F_matrices = []
    for b in range(B):
        A_b = A[b]
        U, S, Vt = torch.svd(A_b)
        F = Vt[-1].view(3, 3)
        U, S, Vt = torch.svd(F)
        S[-1] = 0
        F = U @ torch.diag(S) @ Vt
        F_matrices.append(F)

    return torch.stack(F_matrices)

if __name__ == "__main__":
    points1 = torch.tensor([[[100, 150], [120, 160], [130, 140], [140, 130], [150, 120], [160, 110], [170, 100], [180, 90]]], dtype=torch.float32)
    points2 = torch.tensor([[[110, 140], [130, 150], [140, 130], [150, 120], [160, 110], [170, 100], [180, 90], [190, 80]]], dtype=torch.float32)
    weights = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)

    F = run_8point(points1, points2, weights)
    print(F)