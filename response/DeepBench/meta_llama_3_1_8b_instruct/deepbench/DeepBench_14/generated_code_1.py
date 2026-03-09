import torch
import torch.nn as nn
import torch.nn.functional as F

def do_conv2d(conv, input_, padding=None, weight=None, bias=None):
    if weight is None:
        weight = conv.weight
    if bias is None:
        bias = conv.bias
    if padding is None:
        padding = conv.padding
    return F.conv2d(input_, weight, bias, stride=conv.stride, padding=padding, dilation=conv.dilation, groups=conv.groups)

if __name__ == "__main__":
    # Create a sample convolutional layer
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True)
    conv.weight.data.fill_(1)
    conv.bias.data.fill_(0)
    
    # Create a sample input tensor
    input_ = torch.randn(1, 1, 5, 5)
    
    # Perform the convolution operation
    output = do_conv2d(conv, input_)
    
    # Print the output shape
    print(output.shape)
    
    # Manually perform the convolution operation using F.conv2d
    manual_output = F.conv2d(input_, conv.weight.data, conv.bias.data, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
    
    # Check if the results are equal
    print(output.equal(manual_output))