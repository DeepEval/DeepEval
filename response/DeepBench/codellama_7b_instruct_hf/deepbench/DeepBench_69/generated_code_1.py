import torch
import torch.nn.functional as F

def filter3d(input, kernel, border_type, normalized=False):
    # Apply padding according to the specified mode
    padding = torch.nn.functional.pad(input, mode=border_type)

    # Convolve the padded input with the kernel
    output = F.conv3d(padding, kernel, stride=1, padding=0)

    # Normalize the kernel if necessary
    if normalized:
        output = output / torch.norm(kernel, p=1, dim=0)

    return output

if __name__ == "__main__":
    # Create sample input and kernel values
    input = torch.randn(1, 3, 3, 3, 3)
    kernel = torch.randn(1, 3, 3, 3, 3)

    # Call the filter3d function with different border types
    output_reflect = filter3d(input, kernel, border_type="reflect")
    output_replicate = filter3d(input, kernel, border_type="replicate")
    output_circular = filter3d(input, kernel, border_type="circular")

    # Print the output tensors
    print(output_reflect)
    print(output_replicate)
    print(output_circular)