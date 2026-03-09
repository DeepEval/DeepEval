import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type='constant', normalized=True, padding='same', behaviour='corr'):
    # Import necessary packages
    import torch
    import torch.nn.functional as F

    # Check if kernel shape is correct
    if kernel.shape == (1, kernel.shape[1], kernel.shape[2]):
        kernel = kernel.expand(input.shape[0], -1, -1)
    elif kernel.shape != (input.shape[0], kernel.shape[1], kernel.shape[2]):
        raise ValueError("Kernel shape must be (1, kH, kW) or (B, kH, kW)")

    # Normalize the kernel if needed
    if normalized:
        kernel = kernel / kernel.sum(dim=(1, 2), keepdim=True)

    # Define padding size
    if padding == 'same':
        padding_size = (kernel.shape[2] // 2, kernel.shape[3] // 2)
    elif padding == 'valid':
        padding_size = (0, 0)
    else:
        raise ValueError("Padding mode must be 'same' or 'valid'")

    # Apply padding
    input_padded = F.pad(input, padding_size * 2, mode=border_type)

    # Define convolution mode
    if behaviour == 'corr':
        output = F.conv2d(input_padded, kernel, groups=input.shape[1])
    elif behaviour == 'conv':
        kernel = torch.flip(kernel, [2, 3])
        output = F.conv2d(input_padded, kernel, groups=input.shape[1])
    else:
        raise ValueError("Behaviour must be 'corr' or 'conv'")

    return output

if __name__ == "__main__":
    # Create sample input values
    input_tensor = torch.randn(2, 3, 5, 5)
    kernel = torch.randn(1, 3, 3)

    # Call the function
    output_tensor = filter2d(input_tensor, kernel, border_type='constant', normalized=True, padding='same', behaviour='corr')

    # Print the results
    print(output_tensor)