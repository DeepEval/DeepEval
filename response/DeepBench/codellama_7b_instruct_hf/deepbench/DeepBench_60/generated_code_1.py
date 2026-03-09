import torch
import torch.nn as nn
import torch.nn.functional as F

def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
    # Perform Deformable Convolution v2 or Deformable Convolution based on the presence of a mask
    if mask is not None:
        # Use Deformable Convolution v2
        output = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)
        output = output + mask * offset
    else:
        # Use Deformable Convolution
        output = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, deformable=True)
        output = output + offset
    return output

if __name__ == "__main__":
    # Create sample input values
    input = torch.randn(1, 3, 224, 224)
    offset = torch.randn(1, 3, 224, 224)
    weight = torch.randn(3, 3, 3, 3)
    bias = torch.randn(3)
    mask = torch.randn(1, 3, 224, 224)

    # Call the function and print the results
    output = deform_conv2d(input, offset, weight, bias=bias, stride=1, padding=0, dilation=1, mask=mask)
    print(output)