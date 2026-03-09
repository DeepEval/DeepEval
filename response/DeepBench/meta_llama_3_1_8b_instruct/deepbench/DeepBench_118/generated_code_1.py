import torch

def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    
    eps = 1e-8
    a = x[..., 0]
    b = x[..., 1]
    c = x[..., 2]
    
    return torch.stack((c+eps, -b, a), dim=-1) - \
           torch.stack((-c, a, b), dim=-1)

if __name__ == "__main__":
    # Create a sample input tensor
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Call the function
    result = cross_product_matrix(x)
    
    # Print the result
    print(result)
    
    # Create another sample input tensor to check error handling
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Call the function with an error
    try:
        cross_product_matrix(y)
    except AssertionError as e:
        print(e)