import torch
import torch.nn as nn

def do_conv2d(conv, input_, padding=None, weight=None, bias=None):
    if weight is None:
        weight = conv.weight
    if bias is None:
        bias = conv.bias
    if padding is None:
        padding = conv.padding

    # Apply convolution operation
    output = torch.nn.functional.conv2d(input_, weight, bias=bias, padding=padding)

    return output

if __name__ == "__main__":
    # Example usage
    # Create sample input values
    input_ = torch.randn(1, 3, 224, 224)

    # Create a convolutional layer with a pre-defined weight and bias
    conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
    conv.weight = torch.randn(64, 3, 3, 3)
    conv.bias = torch.randn(64)

    # Call the do_conv2d function with the sample input values and the convolutional layer
    output = do_conv2d(conv, input_)

    # Print the output
    print(output)