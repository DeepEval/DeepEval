import torch

def get_sobel_kernel2d(device=None, dtype=None):
    # Create the Sobel kernel for x-direction
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=dtype, device=device)

    # Create the Sobel kernel for y-direction
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=dtype, device=device)

    # Stack the kernels horizontally
    sobel_kernel = torch.stack([sobel_x, sobel_y], dim=-1)

    return sobel_kernel

if __name__ == "__main__":
    # Create sample input values
    device = 'cpu'
    dtype = torch.float32

    # Call the function and print the results
    sobel_kernels = get_sobel_kernel2d(device=device, dtype=dtype)
    print(sobel_kernels)