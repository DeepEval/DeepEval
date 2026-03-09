import torch
import numpy as np

def run_7point(points1, points2):
    assert points1.shape == points2.shape, "Input tensors must have the same shape."
    assert points1.shape[1] == 7, "Each batch must contain exactly 7 points."
    B, N, _ = points1.shape
    
    # Normalize points
    def normalize_points(points):
        centroid = points.mean(dim=1, keepdim=True)
        points_centered = points - centroid
        mean_dist = points_centered.norm(dim=2).mean(dim=1, keepdim=True)
        scale = np.sqrt(2) / mean_dist
        T = torch.eye(3, device=points.device, dtype=points.dtype)
        T[0:2, 0:2] = torch.diag(torch.tensor([scale], device=points.device, dtype=points.dtype))
        T[0:2, 2:3] = centroid[:, :, 0:1] * scale
        return T, points_centered
    
    T1, points1_norm = normalize_points(points1)
    T2, points2_norm = normalize_points(points2)
    
    # Construct the linear system
    A = []
    for i in range(N):
        x1, y1 = points1_norm[:, i, 0], points1_norm[:, i, 1]
        x2, y2 = points2_norm[:, i, 0], points2_norm[:, i, 1]
        A.append(torch.stack([
            x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, torch.ones_like(x1)
        ], dim=0))
    A = torch.stack(A, dim=0).view(N, 9, N)
    
    # Solve using SVD
    U, S, Vt = torch.linalg.svd(A)
    F_svd = Vt[-1].view(9)
    
    # Enforce rank 2 constraint
    U, S, Vt = torch.linalg.svd(F_svd.view(3, 3))
    S[2] = 0
    F_svd = (U * S.unsqueeze(1)) @ Vt
    
    # Compute the cubic polynomial
    F1 = F_svd.view(3, 3)
    eigvals = torch.linalg.eigvals(F1)
    coeffs = torch.poly(eigvals.real)
    
    # Solve for roots
    roots = torch.roots(coeffs.flip(0))
    roots = roots[torch.isreal(roots) & (roots.abs() <= 1)]
    
    # Compute potential fundamental matrices
    potential_Fs = []
    for root in roots:
        F = root.item() * F1
        U, S, Vt = torch.linalg.svd(F)
        S[-1] = 0
        F = (U @ S.diag_embed()) @ Vt
        potential_Fs.append(F.view(1, 3, 3))
    
    potential_Fs = torch.stack(potential_Fs, dim=0).view(B, -1, 3, 3)
    return potential_Fs

if __name__ == "__main__":
    points1 = torch.tensor([
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
    ])
    points2 = torch.tensor([
        [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
        [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    ])
    Fs = run_7point(points1, points2)
    print(Fs)