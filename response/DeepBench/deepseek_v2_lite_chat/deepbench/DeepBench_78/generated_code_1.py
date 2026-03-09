import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type='replicate', padding='same', behaviour='corr', normalized=False, mode='conv'):
    # Convert kernel to 2D if it's 3D
    if kernel.ndim == 3:
        kernel = kernel.unsqueeze(0)
    if kernel.size(0) != input.size(1):
        raise ValueError("Kernel dimensions do not match input channels")
    
    # Determine the padding size
    if border_type == 'constant':
        padding_size = (1, 1)
    elif border_type == 'reflect':
        padding_size = (0, 0)
    elif border_type == 'replicate':
        padding_size = (1, 1)
    elif border_type == 'circular':
        padding_size = (1, 1)
    else:
        raise ValueError("Unsupported border type")
    
    # Create padding tensor
    if border_type != 'valid' and padding == 'same':
        if behaviour == 'corr':
            padding = F.pad(input, (0, 0, 0, 0), mode='reflect')
        else:
            padding = F.pad(input, (0, 0, 0, 0), mode='replicate')
    
    # Check behaviour and mode
    if behaviour == 'corr':
        kernel = kernel / (kernel.abs().mean() + 1e-5)
    if mode == 'corr':
        output = F.conv2d(padding, kernel, padding=padding_size)
    else:
        output = F.conv3d(padding, kernel, padding=padding_size)
    
    if normalized:
        output = output.abs() / output.abs().sum()
    
    return output

if __name__ == "__main__":
    # Create a sample input tensor
    input_tensor = torch.randn(1, 3, 16, 16)
    
    # Create a sample kernel tensor
    kernel_tensor = torch.randn(1, 3, 3, 3)
    
    # Call the function
    convolved_tensor = filter2d(input_tensor, kernel_tensor, behaviour='corr')
    
    # Print the results
    print(convolved_tensor.shape)