import torch

def run_8point(points1, points2, weights):
    B, N, _ = points1.shape
    
    # Construct matrix A using the point correspondences
    x1 = points1[..., 0]
    y1 = points1[..., 1]
    x2 = points2[..., 0]
    y2 = points2[..., 1]

    A = torch.stack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, torch.ones_like(x1)], dim=-1)
    
    # Apply weights
    W = weights.unsqueeze(-1)
    A_weighted = A * W

    # Solve the weighted least squares problem using SVD
    U, S, V = torch.svd(A_weighted)
    F = V[..., -1].reshape(B, 3, 3)

    # Enforce the rank 2 constraint on F by setting the last singular value to 0
    U_f, S_f, V_f = torch.svd(F)
    S_f[..., -1] = 0
    F_rank2 = U_f @ torch.diag_embed(S_f) @ V_f.transpose(-2, -1)

    return F_rank2

if __name__ == "__main__":
    B, N = 1, 8  # For a minimal example, use batch size 1 and 8 points
    points1 = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                             [0.5, 0.5], [0.5, 0.0], [0.0, 0.5], [1.0, 0.5]]], dtype=torch.float32)
    points2 = torch.tensor([[[0.1, 0.0], [1.1, 0.0], [1.1, 1.1], [0.1, 1.1],
                             [0.6, 0.5], [0.6, 0.0], [0.1, 0.5], [1.1, 0.5]]], dtype=torch.float32)
    weights = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

    F = run_8point(points1, points2, weights)
    print(F)