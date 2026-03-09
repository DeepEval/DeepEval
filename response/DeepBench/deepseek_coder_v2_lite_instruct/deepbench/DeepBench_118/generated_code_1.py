import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 3:
        raise AssertionError(f"Input tensor must have a last dimension of 3, but got {x.shape[-1]}")
    
    zeros = torch.zeros_like(x[..., 0:1])
    return torch.stack([
        torch.stack([zeros, -x[..., 2], x[..., 1]], dim=-1),
        torch.stack([x[..., 2], zeros, -x[..., 0]], dim=-1),
        torch.stack([-x[..., 1], x[..., 0], zeros], dim=-1)
    ], dim=-2)

if __name__ == "__main__":
    # Example usage
    vec = torch.tensor([1.0, 2.0, 3.0])
    cp_matrix = cross_product_matrix(vec)
    print("Cross-product matrix for vector", vec, "is:\n", cp_matrix)

    # Testing with multiple vectors
    vectors = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cp_matrices = cross_product_matrix(vectors)
    print("Cross-product matrices for vectors:\n", vectors, "are:\n", cp_matrices)