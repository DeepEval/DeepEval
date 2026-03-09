import numpy as np
import torch

def _get_splat_kernel_normalization(offsets, sigma=0.5):
    if sigma <= 0.0:
        raise ValueError("Sigma must be greater than 0.0")

    normalization_constant = torch.exp(-np.sum(np.square(offsets)) / (2 * np.square(sigma))) + 0.05
    return (1 + 0.05) / normalization_constant

if __name__ == "__main__":
    # Create sample input values
    offsets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = 0.5

    # Call the function and print the results
    result = _get_splat_kernel_normalization(offsets, sigma)
    print("Normalization constant: ", result)