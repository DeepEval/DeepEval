import torch

def get_sobel_kernel2d(device=None, dtype=None):
    sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype, device=device)
    sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype, device=device)
    
    sobel_kernels = torch.stack([sobel_kernel_x, sobel_kernel_y])
    return sobel_kernels

if __name__ == "__main__":
    # Example usage
    device = torch.device('cpu')  # You can change this to 'cuda' if you have a GPU
    dtype = torch.float32  # You can change this to any suitable dtype

    sobel_kernels = get_sobel_kernel2d(device=device, dtype=dtype)
    print("Sobel Kernel for X and Y directions:")
    print(sobel_kernels)