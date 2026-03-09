import torch
import torch.nn as nn

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError("sigma must be greater than 0.0")
    
    offset_square_sum = offsets.pow(2).sum()
    normalization_constant = torch.exp(-offset_square_sum / (2 * sigma))
    epsilon = 0.05
    normalization_constant += epsilon
    
    normalized_value = (1 + epsilon) / normalization_constant
    
    return normalized_value

if __name__ == "__main__":
    # Sample input values
    offsets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    sigma = 0.5
    
    # Call the function and print the results
    normalized_value = _get_splat_kernel_normalization(offsets, sigma)
    print("Normalized Value:", normalized_value)