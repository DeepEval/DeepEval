import torch
import torch.nn.functional as F

def filter3d(input, kernel, border_type='reflect', normalized=True):
    # Normalize the kernel if required
    if normalized:
        kernel = kernel / kernel.sum()
    
    # Apply padding to the input tensor
    if border_type == 'reflect':
        padding = [kernel.shape[i] // 2 for i in range(1, len(kernel.shape))]
        input = F.pad(input, padding, mode='reflect')
    elif border_type == 'replicate':
        padding = [kernel.shape[i] // 2 for i in range(1, len(kernel.shape))]
        input = F.pad(input, padding, mode='replicate')
    elif border_type == 'constant':
        padding = [kernel.shape[i] // 2 for i in range(1, len(kernel.shape))]
        input = F.pad(input, padding, mode='constant', value=0)
    
    # Perform the convolution
    output = F.conv3d(input, kernel, padding=0)
    
    return output

if __name__ == "__main__":
    # Create a sample input tensor
    input = torch.randn(2, 3, 4, 5, 6)
    
    # Create a sample kernel
    kernel = torch.randn(2, 3, 3, 3, 3)
    
    # Call the function
    output = filter3d(input, kernel)
    
    # Print the results
    print(output.shape)