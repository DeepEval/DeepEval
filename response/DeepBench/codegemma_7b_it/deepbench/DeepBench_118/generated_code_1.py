import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)

    return torch.stack(
        [
            torch.stack([-x[:, 2], x[:, 1], x[:, 0]], dim=-1),
            torch.stack([-x[:, 0], x[:, 2], x[:, 1]], dim=-1),
            torch.stack([-x[:, 1], x[:, 0], x[:, 2]], dim=-1),
        ],
        dim=-2,
    )


if __name__ == "__main__":
    x = torch.randn(3, 3)
    cp_matrix = cross_product_matrix(x)
    print(cp_matrix)