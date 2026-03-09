import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError('Invalid sigma value, must be greater than 0.0.')
    normalization_constant = torch.sum(torch.exp(-(offsets.pow(2) / (2 * sigma ** 2))), dim=1)
    return (1 + 0.05) / (normalization_constant + 0.05)

if __name__ == "__main__":
    offsets = torch.tensor([[1.0, 2.0], [-3.0, -4.0]])
    sigma = 0.75

    normalization = _get_splat_kernel_normalization(offsets, sigma)

    print(normalization)