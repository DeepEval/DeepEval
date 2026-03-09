import torch
import numpy as np

def find_homography_lines_dlt(ls1, ls2, weights=None):
    B, N, _, _ = ls1.shape
    # Convert line segments to point correspondences
    p1 = torch.cat([ls1[:, :, 0, :], torch.ones((B, N, 1))], dim=2)
    p2 = torch.cat([ls2[:, :, 0, :], torch.ones((B, N, 1))], dim=2)
    
    # Initialize matrix A and vector b for linear system
    A = torch.zeros((B * 4, 9))
    b = torch.zeros((B * 4))
    
    for i in range(B):
        if weights is None:
            weight = 1
        else:
            weight = weights[i]
        
        # Compute the partial derivatives of the error function with respect to the homography parameters
        for j in range(N):
            A1 = torch.tensor([[p2[i, j, 0], p2[i, j, 1], 1, 0, 0, 0, -p1[i, j, 0] * p2[i, j, 0], -p1[i, j, 1] * p2[i, j, 0], -p2[i, j, 0]]])
            A2 = torch.tensor([[0, 0, 0, p2[i, j, 0], p2[i, j, 1], 1, -p1[i, j, 0] * p2[i, j, 1], -p1[i, j, 1] * p2[i, j, 1], -p2[i, j, 1]]])
            A = torch.cat([A, weight * A1, weight * A2], dim=0)
            b = torch.cat([b, weight * 0], dim=0)
            
    # Solve the linear system using Weighted Least Squares
    H = torch.linalg.lstsq(A, b, rcond=None)[0].view(B, 3, 3)
    
    return H

if __name__ == "__main__":
    # Generate sample line segments
    B, N = 2, 4
    ls1 = torch.rand((B, N, 2, 2))
    ls2 = torch.rand((B, N, 2, 2))
    
    # Compute homography matrix
    H = find_homography_lines_dlt(ls1, ls2)
    
    # Print the results
    print(H)