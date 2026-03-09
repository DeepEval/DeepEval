import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type='constant', normalized=False, padding='same', behaviour='corr'):
    B, C, H, W = input.shape
    kH, kW = kernel.shape[1], kernel.shape[2]
    
    if padding == 'same':
        pad = (kW // 2, kW // 2, kH // 2, kH // 2)
    elif padding == 'valid':
        pad = (0, 0, 0, 0)
    else:
        raise ValueError("Invalid padding type. Choose 'same' or 'valid'.")

    if border_type == 'constant':
        input_padded = F.pad(input, pad, mode='constant', value=0)
    elif border_type == 'reflect':
        input_padded = F.pad(input, pad, mode='reflect')
    elif border_type == 'replicate':
        input_padded = F.pad(input, pad, mode='replicate')
    elif border_type == 'circular':
        input_padded = F.pad(input, pad, mode='circular')
    else:
        raise ValueError("Invalid border type. Choose 'constant', 'reflect', 'replicate' or 'circular'.")

    if normalized:
        kernel = kernel / kernel.sum()

    if behaviour == 'conv':
        kernel = torch.flip(kernel, [1, 2])  # Flip the kernel for true convolution

    output = F.conv2d(input_padded, kernel, stride=1, padding=0, groups=B if kernel.shape[0] == B else 1)
    
    return output

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 5, 5)  # Batch size of 1, 3 channels, 5x5 image
    kernel = torch.tensor([[[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]]], dtype=torch.float32)  # Example kernel (1, 3, 3)

    output_tensor = filter2d(input_tensor, kernel, border_type='constant', normalized=False, padding='same', behaviour='corr')
    print("Output Tensor Shape:", output_tensor.shape)
    print("Output Tensor:", output_tensor)