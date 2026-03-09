import numpy as np
import torch

def xavier_normal_(tensor, gain=1.0, generator=None):
    # Sample from a Xavier normal distribution
    std = gain * np.sqrt(2 / (tensor.shape[0] + tensor.shape[1]))
    samples = torch.normal(0, std, size=tensor.shape)

    # Assign the samples to the input tensor
    tensor.data.copy_(samples)

if __name__ == "__main__":
    # Create a sample input tensor
    tensor = torch.randn(2, 3)

    # Call the function and print the results
    xavier_normal_(tensor)
    print(tensor)