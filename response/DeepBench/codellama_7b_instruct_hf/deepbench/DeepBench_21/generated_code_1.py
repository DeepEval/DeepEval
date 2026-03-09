import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError("Sigma must be greater than 0.0")
    epsilon = 0.05
    normalization = torch.exp(-(offsets**2) / (2 * sigma**2))
    normalization = normalization.sum() + epsilon
    return (1 + epsilon) / normalization

if __name__ == "__main__":
    offsets = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = 0.5
    norm = _get_splat_kernel_normalization(offsets, sigma)
    print("Normalization value:", norm.item())