import torch
import torch.nn.functional as F

def filter3d(input, kernel, border_type='zeros', normalized=False):
    B, C, D, H, W = input.shape
    kernel_size = kernel.shape

    if normalized:
        kernel = kernel / kernel.sum()

    padding = [(k - 1) // 2 for k in kernel_size]
    input_padded = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]), mode=border_type)

    output = torch.empty_like(input)
    for b in range(B):
        for c in range(C):
            output[b, c] = F.conv3d(input_padded[b, c].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    
    return output

if __name__ == "__main__":
    B, C, D, H, W = 1, 1, 5, 5, 5  # Example dimensions
    input_tensor = torch.rand((B, C, D, H, W))
    kernel = torch.ones((3, 3, 3))
    
    convolved_output = filter3d(input_tensor, kernel, border_type='zeros', normalized=True)
    print("Input Tensor:")
    print(input_tensor)
    print("\nKernel:")
    print(kernel)
    print("\nConvolved Output:")
    print(convolved_output)