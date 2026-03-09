import torch
import numpy as np
from typing import Tuple

class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int,
  int], stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases randomly
        self.weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.bias = torch.randn(out_channels)

    # Other methods and functionalities of the Conv2d class here

if __name__ == "__main__":
    # Example test case:
    conv_layer = Conv2d(3, 64, (3, 3), stride=(2, 2), padding=(1, 1))

    # Create input tensor
    input_tensor = torch.randn(1, 3, 224, 224)

    # Perform convolution
    output_tensor = conv_layer(input_tensor)

    # Print output tensor shape
    print(output_tensor.shape)