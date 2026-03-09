import torch

def find_homography_lines_dlt(ls1, ls2, weights=None):
    B, N, _, _ = ls1.shape
    A = torch.zeros((B, N * 2, 9), dtype=ls1.dtype, device=ls1.device)

    for i in range(N):
        l1 = ls1[:, i]
        l2 = ls2[:, i]
        
        A[:, i * 2] = torch.stack([
            l1[:, 0] * l2[:, 0],
            l1[:, 1] * l2[:, 0],
            l2[:, 0],
            l1[:, 0] * l2[:, 1],
            l1[:, 1] * l2[:, 1],
            l2[:, 1],
            l1[:, 0],
            l1[:, 1],
            torch.ones_like(l1[:, 0])
        ], dim=1).view(B, 9)

        A[:, i * 2 + 1] = torch.stack([
            l1[:, 0] * l2[:, 1],
            l1[:, 1] * l2[:, 1],
            l2[:, 1],
            l1[:, 0] * l2[:, 0],
            l1[:, 1] * l2[:, 0],
            l2[:, 0],
            l1[:, 0],
            l1[:, 1],
            torch.ones_like(l1[:, 0])
        ], dim=1).view(B, 9)

    if weights is not None:
        W = torch.diag_embed(weights)
        A = torch.bmm(W, A.view(B, N * 2, 9)).view(B, N * 2, 9)

    U, S, Vh = torch.linalg.svd(A)
    H = Vh[:, -1].view(B, 3, 3)

    return H

if __name__ == "__main__":
    B = 1
    N = 4
    ls1 = torch.tensor([[[0, 0], [1, 1]], 
                         [[1, 0], [0, 1]], 
                         [[0, 1], [1, 0]], 
                         [[1, 1], [1, 0]]], dtype=torch.float32).unsqueeze(0)
    
    ls2 = torch.tensor([[[0, 0], [2, 2]], 
                         [[2, 0], [0, 2]], 
                         [[0, 2], [2, 0]], 
                         [[2, 2], [2, 0]]], dtype=torch.float32).unsqueeze(0)

    weights = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)
    
    H = find_homography_lines_dlt(ls1, ls2, weights)
    print(H)