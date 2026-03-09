import torch
import torch.nn as nn

def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
    return layer

def function_name(args):
    return Conv2d(*args)

if __name__ == "__main__":
    # Define some sample input values
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 1
    bias = True

    # Create the Conv2d layer
    conv_layer = function_name((in_channels, out_channels, kernel_size, stride, padding, bias))

    # Print the Conv2d layer
    print(conv_layer)

    # Create a random input tensor
    input_tensor = torch.randn(1, in_channels, 28, 28)

    # Print the output of the Conv2d layer
    output = conv_layer(input_tensor)
    print(output.shape)