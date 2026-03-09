import torch

def get_sobel_kernel2d(device=None, dtype=None):
    kernels = torch.zeros(2, 3, 3, device=device, dtype=dtype)
    kernels[0] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
    kernels[1] = torch.tensor([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], device=device, dtype=dtype)
    return kernels

if __name__ == "__main__":
    # Sample input values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Call the function and print the results
    sobel_kernels = get_sobel_kernel2d(device=device, dtype=dtype)
    print("Sobel kernels for x direction:", sobel_kernels[0])
    print("Sobel kernels for y direction:", sobel_kernels[1])