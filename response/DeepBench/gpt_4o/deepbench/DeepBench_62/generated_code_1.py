import torch

def get_sobel_kernel_5x5_2nd_order():
    gxx = torch.tensor([
        [1, 4, 6, 4, 1],
        [2, 8, 12, 8, 2],
        [0, 0, 0, 0, 0],
        [-2, -8, -12, -8, -2],
        [-1, -4, -6, -4, -1]
    ], dtype=torch.float32)
    return gxx

def _get_sobel_kernel_5x5_2nd_order_xy():
    gxy = torch.tensor([
        [-1, -2, 0, 2, 1],
        [-4, -8, 0, 8, 4],
        [-6, -12, 0, 12, 6],
        [-4, -8, 0, 8, 4],
        [-1, -2, 0, 2, 1]
    ], dtype=torch.float32)
    return gxy

def get_sobel_kernel2d_2nd_order(device='cpu', dtype=torch.float32):
    gxx = get_sobel_kernel_5x5_2nd_order().to(device=device, dtype=dtype)
    gyy = gxx.t().to(device=device, dtype=dtype)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy().to(device=device, dtype=dtype)
    return torch.stack((gxx, gxy, gyy), dim=0)

if __name__ == "__main__":
    device = 'cpu'
    dtype = torch.float32
    kernels = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    print(kernels)