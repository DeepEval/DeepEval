import torch
import torch.nn as nn

def do_conv2d(conv, input_, padding=None, weight=None, bias=None):
    if padding is None:
        padding = conv.padding
    if weight is None:
        weight = conv.weight
    if bias is None:
        bias = conv.bias

    return nn.functional.conv2d(input_, weight, bias, stride=conv.stride, padding=padding)

if __name__ == "__main__":
    # Create a sample convolutional layer
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    
    # Create a sample input tensor (1 sample, 1 channel, 5x5 image)
    input_tensor = torch.randn(1, 1, 5, 5)

    # Call the do_conv2d function
    output_tensor = do_conv2d(conv_layer, input_tensor)

    # Print the results
    print("Input Tensor:")
    print(input_tensor)
    print("Output Tensor:")
    print(output_tensor)