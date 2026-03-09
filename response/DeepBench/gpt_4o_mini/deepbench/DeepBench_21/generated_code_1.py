import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError("Sigma must be greater than 0.0.")
    
    normalization_constant = torch.sum(torch.exp(-torch.pow(offsets, 2) / (2 * sigma ** 2)))
    epsilon = 0.05
    normalized_value = (1 + epsilon) / normalization_constant
    
    return normalized_value

if __name__ == "__main__":
    offsets = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = 0.5
    result = _get_splat_kernel_normalization(offsets, sigma)
    print(result)