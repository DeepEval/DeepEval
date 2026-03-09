import numpy as np
import torch
import torch.nn.functional as F

def filter3d(input, kernel, border_type='zeros', normalized=False):
    B, C, D, H, W = input.shape
    kD, kH, kW = kernel.shape
    
    if normalized:
        kernel = kernel / kernel.sum()
    
    if border_type == 'zeros':
        padding = (kD // 2, kH // 2, kW // 2)
        input_padded = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]), mode='constant', value=0)
    elif border_type == 'reflect':
        padding = (kD // 2, kH // 2, kW // 2)
        input_padded = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]), mode='reflect')
    elif border_type == 'replicate':
        padding = (kD // 2, kH // 2, kW // 2)
        input_padded = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]), mode='replicate')
    else:
        raise ValueError("Unsupported border_type")

    output = F.conv3d(input_padded, kernel.view(1, C, 1, kD, kH, kW), stride=1, padding=0)

    return output

if __name__ == "__main__":
    B, C, D, H, W = 1, 1, 5, 5, 5
    input_tensor = torch.rand(B, C, D, H, W)
    kernel = torch.rand(1, 1, 3, 3, 3)  # 3D kernel with shape (out_channels, in_channels, depth, height, width)
    convolved_tensor = filter3d(input_tensor, kernel, border_type='zeros', normalized=True)
    print(convolved_tensor)