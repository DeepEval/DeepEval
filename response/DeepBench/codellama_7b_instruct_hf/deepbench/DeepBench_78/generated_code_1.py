import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type, normalized, padding, behaviour):
    # Validate input arguments
    assert isinstance(input, torch.Tensor), "input must be a torch.Tensor"
    assert isinstance(kernel, torch.Tensor), "kernel must be a torch.Tensor"
    assert border_type in ['constant', 'reflect', 'replicate', 'circular'], "invalid border_type"
    assert padding in ['same', 'valid'], "invalid padding"
    assert behaviour in ['corr', 'conv'], "invalid behaviour"

    # Apply padding
    if padding == 'same':
        input = F.pad(input, (kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2), border_type)
    elif padding == 'valid':
        input = F.pad(input, (kernel.shape[2] // 2, kernel.shape[2] // 2, kernel.shape[3] // 2, kernel.shape[3] // 2), border_type, value=0)

    # Apply kernel convolution
    if behaviour == 'corr':
        output = F.conv2d(input, kernel, padding=0, groups=input.shape[1])
    elif behaviour == 'conv':
        kernel = kernel.flip(dims=(2, 3))
        output = F.conv2d(input, kernel, padding=0, groups=input.shape[1])

    # Normalize kernel
    if normalized:
        kernel = kernel / kernel.sum()

    return output

if __name__ == "__main__":
    # Example usage
    input = torch.randn(1, 3, 4, 4)
    kernel = torch.randn(1, 3, 3, 3)
    output = filter2d(input, kernel, border_type='constant', normalized=True, padding='same', behaviour='corr')
    print(output.shape)

    # Output: torch.Size([1, 3, 4, 4])