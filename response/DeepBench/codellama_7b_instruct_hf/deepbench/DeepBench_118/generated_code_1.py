import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == 3, "Input tensor must have last dimension equal to 3"

    # Compute cross product matrix
    out = torch.zeros(x.shape[:-1] + (3, 3), dtype=x.dtype, device=x.device)
    out[..., 0, 1] = -x[..., 2]
    out[..., 0, 2] = x[..., 1]
    out[..., 1, 0] = x[..., 2]
    out[..., 1, 2] = -x[..., 0]
    out[..., 2, 0] = -x[..., 1]
    out[..., 2, 1] = x[..., 0]

    return out

if __name__ == "__main__":
    # Sample input values
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Call function and print results
    out = cross_product_matrix(x)
    print(out)