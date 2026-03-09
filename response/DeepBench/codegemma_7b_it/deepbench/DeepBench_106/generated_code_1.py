import torch

def find_homography_lines_dlt(ls1, ls2, weight=None):
    B, N, *_ = ls1.shape
    if weight is not None:
        weight = weight.reshape(B, N, 1, 1)

    x1 = torch.cat([ls1[:, :, 0, :], torch.ones((B, N, 1))], dim=-1)
    x2 = torch.cat([ls2[:, :, 0, :], torch.ones((B, N, 1))], dim=-1)
    y1 = torch.cat([ls1[:, :, 1, :], torch.ones((B, N, 1))], dim=-1)
    y2 = torch.cat([ls2[:, :, 1, :], torch.ones((B, N, 1))], dim=-1)

    A = torch.cat([x1 * y2 - y1 * x2, -x1, y1, -y2], dim=1).reshape(B, N, 2, 2)

    if weight is not None:
        A = A.view(B * N, 2, 2)
        weight = weight.view(B * N, 1, 1)
        A = A.transpose(1, 2) @ weight @ A
    else:
        A = A.transpose(1, 2) @ A

    b = torch.cat([y2 - x1 * y1, -y1], dim=1).reshape(B, N, 2, 1)
    if weight is not None:
        b = b.view(B * N, 2, 1)
        b = b.transpose(1, 2) @ weight @ b

    H = torch.linalg.svd(A).Vh.transpose(1, 2).reshape(B, N, 3, 3)

    return H[:, 0] / H[:, 0, 2, 2]

if __name__ == "__main__":
    # Example usage
    ls1 = torch.randn(2, 5, 2, 2)
    ls2 = torch.randn(2, 5, 2, 2)

    H = find_homography_lines_dlt(ls1, ls2)

    print(H)