import torch
import torch.nn as nn
import torch.nn.functional as F

def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, mask=None):
    if mask is not None:
        return deformable_conv2d(input, offset, weight, bias, stride, padding, dilation, groups, mask)
    else:
        return deformable_conv2d(input, offset, weight, bias, stride, padding, dilation, groups)

def deformable_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, mask=None):
    B, C, H, W = input.shape
    M, K, _, _ = weight.shape
    B, _, HW = offset.shape

    if B!= 1:
        raise ValueError('Batch size should be 1')
    
    if mask is not None:
        mask = mask.view(1, 1, H, W)
        offset = offset.view(1, K, H, W)

        input = input.unsqueeze(0).repeat(1, M, 1, 1, 1)
        weight = weight.view(1, C, M, K, 1, 1)
        mask = mask.repeat(1, 1, 1, 1, M, 1)
        offset = offset.repeat(1, 1, 1, 1, M)
        
        x = F.grid_sample(input, offset - mask * 2, padding_mode='zeros', align_corners=False)
        x = x.view(B * M, C, H, W)
    else:
        input = input.unsqueeze(1).repeat(1, M, 1, 1, 1)
        weight = weight.view(1, C, M, K, 1, 1)

        x = F.grid_sample(input, offset, padding_mode='zeros', align_corners=False)
        x = x.view(B * M, C, H, W)

    x = x * weight
    if bias is not None:
        x = x + bias
    if groups == 1:
        x = x.view(B, C * M, H, W).sum(dim=1)
    else:
        x = x.view(B, C, M, H, W).sum(dim=2)
    x = F.conv2d(x, weight=None, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return x

if __name__ == "__main__":
    # Create sample input values
    B, C, H, W = 1, 3, 10, 10
    input = torch.randn(B, C, H, W)
    offset = torch.randn(1, 3, H, W)
    weight = torch.randn(1, 3, 3, 3, 1, 1)
    bias = torch.randn(1, 3, 3, 3, 1, 1)

    # Create a mask for deformable convolution
    mask = torch.randint(0, 2, (1, 1, H, W))
    mask = mask.float()

    # Call the function
    result = deform_conv2d(input, offset, weight, bias, stride=1, padding=0, dilation=1, groups=1, mask=mask)
    print(result.shape)

    # Call the function without a mask
    result = deform_conv2d(input, offset, weight, bias, stride=1, padding=0, dilation=1, groups=1)
    print(result.shape)