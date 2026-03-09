import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    i, j, k = torch.eye(3), torch.zeros_like(torch.eye(3)), torch.zeros_like(torch.eye(3))
    i[:, 1, 2] = -1
    j[:, 0, 2] = -1
    k[:, 0, 1] = -1
    return torch.stack([i * x[..., None], j * x[..., None], k * x[..., None]], dim=-2)

if __name__ == "__main__":
    x = torch.randn(2, 3)
    result = cross_product_matrix(x)
    print(result)