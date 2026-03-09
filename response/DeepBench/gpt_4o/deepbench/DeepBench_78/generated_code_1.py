import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type='constant', normalized=False, padding='same', behaviour='corr'):
    B, C, H, W = input.shape
    kB, kH, kW = kernel.shape

    if normalized:
        kernel = kernel / kernel.abs().sum(dim=(-2, -1), keepdim=True)

    if padding == 'same':
        pad_h = (kH - 1) // 2
        pad_w = (kW - 1) // 2
        padding = [pad_w, pad_w, pad_h, pad_h]
    elif padding == 'valid':
        padding = [0, 0, 0, 0]

    if behaviour == 'conv':
        kernel = kernel.flip(-1).flip(-2)

    kernel = kernel.expand(B, C, kH, kW)

    input_padded = F.pad(input, padding, mode=border_type)
    output = F.conv2d(input_padded, kernel, groups=C)

    return output

if __name__ == "__main__":
    B, C, H, W = 1, 1, 5, 5
    kH, kW = 3, 3
    
    input_tensor = torch.arange(B * C * H * W, dtype=torch.float32).view(B, C, H, W)
    kernel = torch.tensor([[[1, 0, -1], [1, 0, -1], [1, 0, -1]]], dtype=torch.float32)

    output = filter2d(input_tensor, kernel, border_type='reflect', normalized=False, padding='same', behaviour='corr')
    print(output)