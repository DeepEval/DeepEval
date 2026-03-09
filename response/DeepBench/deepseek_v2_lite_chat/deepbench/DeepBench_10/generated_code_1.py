import torch
import math

def silu(input_tensor, inplace=False):
    # Calculate SiLU function element-wise
    if inplace:
        input_tensor = input_tensor.add_(input_tensor.mul(input_tensor.sigmoid()))
    else:
        input_tensor = input_tensor.sigmoid().mul(input_tensor)
    return input_tensor

if __name__ == "__main__":
    # Example usage
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    result = silu(input_tensor, inplace=False)
    print(result)  # Output should be [0.7310586, 1.7012577, 3.5406196]

    input_tensor_inplace = torch.tensor([1.0, 2.0, 3.0])
    silu_inplace = silu(input_tensor_inplace, inplace=True)
    print(silu_inplace)  # Output should be [0.7310586, 1.7012577, 3.5406196]

    # Check if results are as expected
    assert torch.allclose(result, [0.7310586, 1.7012577, 3.5406196]), "Test failed"
    assert torch.allclose(silu_inplace, [0.7310586, 1.7012577, 3.5406196]), "Test failed"
    print("All tests passed")