import torch
import torch.nn.functional as F

def filter3d(input_tensor, kernel, border_type='reflect', normalized=False):
    """
    Convolves a 3D kernel with a given input tensor.

    Parameters:
    input_tensor (torch.Tensor): The input tensor with shape (B, C, D, H, W).
    kernel (torch.Tensor): The kernel tensor with shape (C, K, K, K).
    border_type (str): The padding mode to be applied ('reflect', 'replicate', 'circular', 'edge', 'constant').
    normalized (bool): If True, the kernel is L1 normalized.

    Returns:
    torch.Tensor: The convolved tensor of the same size and number of channels as the input tensor.
    """
    # Ensure the kernel is 3D
    kernel = kernel.unsqueeze(0)
    # Convolve using F.conv3d with specified padding and normalization
    output = F.conv3d(input_tensor, kernel, padding=border_type, groups=input_tensor.size(1))
    # Apply normalization if requested
    if normalized:
        kernel = kernel.pow(1).sum(2).sum(2).sum(2)  # L1 normalization
        output = output / kernel.clamp(min=1)
    return output

if __name__ == "__main__":
    # Sample input values
    batch_size = 3
    num_channels = 4
    depth = 5
    height = 6
    width = 7
    kernel_size = 3
    input_tensor = torch.randn(batch_size, num_channels, depth, height, width)
    kernel = torch.randn(num_channels, kernel_size, kernel_size, kernel_size)
    border_type = 'reflect'
    normalized = True

    # Call the function
    output = filter3d(input_tensor, kernel, border_type, normalized)

    # Print the results
    print("Output tensor shape:", output.shape)
    print("Output tensor mean:", output.mean())
    print("Output tensor std:", output.std())