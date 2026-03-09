import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)

    # Extracting individual components of the vector
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]

    # Construct the skew-symmetric matrix for each vector
    zeros = torch.zeros_like(x1)
    result = torch.stack([
        torch.stack([zeros, -x3, x2], dim=-1),
        torch.stack([x3, zeros, -x1], dim=-1),
        torch.stack([-x2, x1, zeros], dim=-1)
    ], dim=-2)

    return result

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Call the function
    result = cross_product_matrix(input_tensor)

    # Print the results
    print(result)