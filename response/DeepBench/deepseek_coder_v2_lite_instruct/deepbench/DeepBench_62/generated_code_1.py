import torch

def get_sobel_kernel_5x5_2nd_order():
    # Define the 5x5 Sobel kernel for the second derivative in the x-direction (gxx)
    gxx = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=torch.float32)
    return gxx

def function_name(device=None, dtype=torch.float32):
    gxx = get_sobel_kernel_5x5_2nd_order().to(device=device, dtype=dtype)
    gyy = gxx.T  # Transpose the kernel for the second derivative in the y-direction (gyy)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy().to(device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])

def _get_sobel_kernel_5x5_2nd_order_xy():
    # Define the 5x5 mixed partial derivative kernel (gxy)
    gxy = torch.tensor([
        [0, 0, -1, 0, 0],
        [0, -2, 0, -2, 0],
        [-1, 0, 8, 0, -1],
        [0, -2, 0, -2, 0],
        [0, 0, -1, 0, 0]
    ], dtype=torch.float32)
    return gxy

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kernels = function_name(device=device)
    print(kernels)