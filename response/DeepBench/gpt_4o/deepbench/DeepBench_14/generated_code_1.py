import torch
import torch.nn as nn

def do_conv2d(conv, input_, padding=None, weight=None, bias=None):
    if padding is None:
        padding = conv.padding
    
    if weight is None:
        weight = conv.weight

    if bias is None:
        bias = conv.bias

    # Apply the convolution operation
    output = nn.functional.conv2d(input_, weight, bias, stride=conv.stride, padding=padding, dilation=conv.dilation)
    
    return output

if __name__ == "__main__":
    # Create a sample convolutional layer
    conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
    
    # Create a sample input tensor with batch size 1, 3 channels, and 5x5 height and width
    input_tensor = torch.randn(1, 3, 5, 5)
    
    # Perform the convolution using the function
    result = do_conv2d(conv_layer, input_tensor)
    
    # Print the results
    print("Input Tensor:")
    print(input_tensor)
    print("Convolution Result:")
    print(result)