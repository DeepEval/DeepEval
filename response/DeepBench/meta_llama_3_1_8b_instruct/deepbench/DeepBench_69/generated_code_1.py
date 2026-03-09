import torch
import torch.nn as nn

def filter3d(input, kernel, border_type='constant', normalized=False):
    # Apply padding according to the specified mode
    if border_type == 'constant':
        padded_input = nn.ReflectionPad3d(1)(input)
    elif border_type =='replicate':
        padded_input = nn.ReplicationPad3d(1)(input)
    elif border_type == 'circular':
        padded_input = nn.ReplicationPad3d(1)(input)
    elif border_type =='reflect':
        padded_input = nn.ReflectionPad3d(1)(input)
    else:
        raise ValueError('Invalid border type')

    # Convolve the kernel with the padded input
    convolved = nn.functional.conv3d(padded_input, kernel, stride=1, padding=0, dilation=1, groups=input.shape[1])

    # Normalize the kernel if required
    if normalized:
        kernel_sum = torch.sum(torch.abs(kernel))
        convolved = convolved / kernel_sum

    return convolved

if __name__ == "__main__":
    # Create sample input values
    B, C, D, H, W = 1, 1, 2, 3, 4
    input_tensor = torch.randn(B, C, D, H, W)
    kernel_tensor = torch.randn(1, C, 3, 3, 3)

    # Call the function and print the results
    result = filter3d(input_tensor, kernel_tensor)
    print(result.shape)
    print(result)