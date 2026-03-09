import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == 3, "The last dimension of the input tensor must be equal to 3"
    # Construct the cross-product matrix (a skew-symmetric matrix) for a given vector
    cross_product_matrix = torch.stack([
        torch.cross(x, torch.tensor([1, 0, 0]), dim=0),
        torch.cross(x, torch.tensor([0, 1, 0]), dim=0),
        torch.cross(x, torch.tensor([0, 0, 1]), dim=0)
    ], dim=0)
    
    # Ensure the matrix has the correct shape
    assert cross_product_matrix.shape == torch.Size([3, 3]), f"Expected cross_product_matrix.shape == torch.Size([3, 3]), but got {cross_product_matrix.shape}"
    
    return cross_product_matrix


if __name__ == "__main__":
    # Example usage
    input_vector = torch.tensor([1.0, 2.0, 3.0])
    output = cross_product_matrix(input_vector)
    print(output)