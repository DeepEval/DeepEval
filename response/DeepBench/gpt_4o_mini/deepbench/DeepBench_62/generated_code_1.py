import torch

def get_sobel_kernel_5x5_2nd_order():
    return torch.tensor([[1, 4, 6, 4, 1],
                         [4, 16, 24, 16, 4],
                         [6, 24, -48, 24, 6],
                         [4, 16, 24, 16, 4],
                         [1, 4, 6, 4, 1]], dtype=torch.float32) / 256

def _get_sobel_kernel_5x5_2nd_order_xy():
    return torch.tensor([[0, 0, 0, 0, 0],
                         [0, -4, -8, -4, 0],
                         [0, -8, 0, 8, 0],
                         [0, 4, 8, 4, 0],
                         [0, 0, 0, 0, 0]], dtype=torch.float32)

def get_sobel_kernel2d_2nd_order(device=None, dtype=None):
    gxx = get_sobel_kernel_5x5_2nd_order().to(device=device, dtype=dtype)
    gyy = gxx.t()
    gxy = _get_sobel_kernel_5x5_2nd_order_xy().to(device=device, dtype=dtype)
    return torch.stack((gxx, gxy, gyy), dim=0)

if __name__ == "__main__":
    device = 'cpu'  # or 'cuda' if you want to use GPU
    dtype = torch.float32
    sobel_kernels = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    print(sobel_kernels)