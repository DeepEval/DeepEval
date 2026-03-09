import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type='constant', normalized=False, padding='same', behaviour='corr'):
    # Normalize the kernel if required
    if normalized:
        kernel = F.normalize(kernel, p=1, dim=-1, keepdim=True)
    
    # Flip the kernel if required
    if behaviour == 'conv':
        kernel = torch.flip(kernel, [-1, -2])
    
    # Get the kernel shape and the input shape
    kH, kW = kernel.shape[-2:]
    B, C, H, W = input.shape
    
    # Get the padding
    if padding =='same':
        padding = kH // 2
    elif padding == 'valid':
        padding = 0
    else:
        raise ValueError("Invalid padding mode. It should be'same' or 'valid'.")
    
    # Apply padding according to the specified mode
    if border_type == 'constant':
        input = F.pad(input, (padding, padding, padding, padding))
    elif border_type =='reflect':
        input = F.pad(input, (padding, padding, padding, padding),'reflect')
    elif border_type =='replicate':
        input = F.pad(input, (padding, padding, padding, padding),'replicate')
    elif border_type == 'circular':
        input = F.pad(input, (padding, padding, padding, padding), 'circular')
    else:
        raise ValueError("Invalid border type. It should be 'constant','reflect','replicate' or 'circular'.")
    
    # Convolve the input with the kernel
    output = F.conv2d(input, kernel, groups=C)
    
    return output

if __name__ == "__main__":
    # Create a sample input tensor
    input_tensor = torch.randn(2, 3, 5, 5)
    
    # Create a sample kernel
    kernel = torch.randn(1, 3, 3)
    
    # Call the function
    output = filter2d(input_tensor, kernel)
    
    # Print the output
    print(output.shape)