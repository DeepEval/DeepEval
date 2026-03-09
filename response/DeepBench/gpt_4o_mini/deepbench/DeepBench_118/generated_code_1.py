import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    
    zeros = torch.zeros_like(x[..., :1])
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    return torch.cat((
        torch.cat((zeros, -x3, x2), dim=-1).unsqueeze(-2),
        torch.cat((x3, zeros, -x1), dim=-1).unsqueeze(-2),
        torch.cat((-x2, x1, zeros), dim=-1).unsqueeze(-2),
    ), dim=-2)

if __name__ == "__main__":
    sample_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = cross_product_matrix(sample_input)
    print(result)