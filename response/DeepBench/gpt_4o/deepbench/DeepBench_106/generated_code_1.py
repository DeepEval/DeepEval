import torch

def find_homography_lines_dlt(ls1, ls2, weights=None):
    B, N, _, _ = ls1.shape
    if weights is None:
        weights = torch.ones((B, N), dtype=torch.float32)

    A = torch.zeros((B, 2 * N, 9), dtype=torch.float32)
    W = weights.view(B, N, 1, 1)

    for i in range(N):
        l1 = ls1[:, i, :, :]
        l2 = ls2[:, i, :, :]
        x1, y1 = l1[:, 0, 0], l1[:, 0, 1]
        x2, y2 = l1[:, 1, 0], l1[:, 1, 1]
        x1p, y1p = l2[:, 0, 0], l2[:, 0, 1]
        x2p, y2p = l2[:, 1, 0], l2[:, 1, 1]

        Ai = torch.stack([
            x1 * x1p, y1 * x1p, x1p,
            x1 * y1p, y1 * y1p, y1p,
            x1, y1, torch.ones_like(x1)
        ], dim=-1)

        A[:, 2 * i, :] = (W[:, i] * Ai).view(B, 9)

        Ai_prime = torch.stack([
            x2 * x2p, y2 * x2p, x2p,
            x2 * y2p, y2 * y2p, y2p,
            x2, y2, torch.ones_like(x2)
        ], dim=-1)

        A[:, 2 * i + 1, :] = (W[:, i] * Ai_prime).view(B, 9)

    _, _, V = torch.svd(A)
    H = V[:, -1].view(B, 3, 3)
    return H

if __name__ == "__main__":
    B = 1
    N = 4
    ls1 = torch.tensor([
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 2.0]],
            [[1.0, 0.0], [2.0, 1.0]],
            [[1.0, 1.0], [2.0, 2.0]]
        ]
    ])
    ls2 = torch.tensor([
        [
            [[0.0, 0.0], [2.0, 2.0]],
            [[0.0, 2.0], [2.0, 4.0]],
            [[2.0, 0.0], [4.0, 2.0]],
            [[2.0, 2.0], [4.0, 4.0]]
        ]
    ])

    weights = torch.tensor([
        [1.0, 1.0, 1.0, 1.0]
    ])

    H = find_homography_lines_dlt(ls1, ls2, weights)
    print("Computed Homography Matrix:")
    print(H)