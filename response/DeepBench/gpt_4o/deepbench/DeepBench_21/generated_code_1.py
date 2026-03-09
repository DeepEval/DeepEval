import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError("Sigma must be greater than 0.0")
    
    squared_offsets = offsets ** 2
    exp_term = torch.exp(-squared_offsets / (2 * sigma ** 2))
    normalization_constant = torch.sum(exp_term)
    
    epsilon = 0.05
    normalized_value = (1 + epsilon) / normalization_constant

    return normalized_value

if __name__ == "__main__":
    # Sample input values
    offsets = torch.tensor([1.0, 2.0, 3.0])
    sigma = 0.5
    
    # Call the function
    normalized_value = _get_splat_kernel_normalization(offsets, sigma)
    
    # Print the result
    print("Normalized value:", normalized_value)