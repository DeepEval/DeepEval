import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError("Sigma must be greater than 0.0")
    
    # Calculate the normalization constant
    squared_offsets = offsets ** 2
    normalization_constant = torch.sum(torch.exp(-squared_offsets / (2 * sigma ** 2)))
    
    # Add epsilon to ensure gradient flow
    epsilon = 0.05
    normalized_value = (1 + epsilon) / normalization_constant
    
    return normalized_value

if __name__ == "__main__":
    # Create sample input values
    offsets = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    sigma = 0.5
    
    # Call the function
    result = _get_splat_kernel_normalization(offsets, sigma)
    
    # Print the results
    print(result)