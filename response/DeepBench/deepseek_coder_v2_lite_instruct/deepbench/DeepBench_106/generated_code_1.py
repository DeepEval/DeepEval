import torch
import numpy as np

def find_homography_lines_dlt(ls1, ls2, weights=None):
    B, N, _, _ = ls1.shape
    device = ls1.device
    
    # Normalize line coordinates to avoid numerical instability
    ls1_norm = ls1 / torch.norm(ls1, dim=3, keepdim=True)
    ls2_norm = ls2 / torch.norm(ls2, dim=3, keepdim=True)
    
    # Prepare the A matrix for DLT
    A = []
    for i in range(N):
        x1, y1 = ls1_norm[:, i, 0, 0], ls1_norm[:, i, 0, 1]
        x2, y2 = ls1_norm[:, i, 1, 0], ls1_norm[:, i, 1, 1]
        x1_, y1_ = ls2_norm[:, i, 0, 0], ls2_norm[:, i, 0, 1]
        x2_, y2_ = ls2_norm[:, i, 1, 0], ls2_norm[:, i, 1, 1]
        
        A.append(torch.stack([
            x1 * x1_ + y1 * y1_, x1 * y1_ - y1 * x1_, x1, y1, 1, 0, 0, 0,
            -x1_ * x1 - y1_ * y1_, -x1_ * y1 + y1_ * x1_, 0, 0, x1_, y1_
        ], dim=0))
        A.append(torch.stack([
            x2 * x2_ + y2 * y2_, x2 * y2_ - y2 * x2_, x2, y2, 1, 0, 0, 0,
            -x2_ * x2 - y2_ * y2_, -x2_ * y2 + y2_ * x2_, 0, 0, x2_, y2_
        ], dim=0))
    
    A = torch.stack(A, dim=1).to(device)
    
    if weights is not None:
        weights = weights.view(B, N, 1, 1).repeat(1, 1, 8, 1).to(device)
        A = A * weights
    
    # Solve the linear system using SVD
    U, S, Vt = torch.linalg.svd(A)
    H_flat = Vt[:, -1, :]
    H = H_flat.view(B, 3, 3)
    
    return H

if __name__ == "__main__":
    # Example usage
    ls1 = torch.tensor([[
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]]
    ]])
    ls2 = torch.tensor([[
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]]
    ]])
    weights = torch.tensor([[
        [1, 1]
    ]])
    
    homography = find_homography_lines_dlt(ls1, ls2, weights)
    print(homography)