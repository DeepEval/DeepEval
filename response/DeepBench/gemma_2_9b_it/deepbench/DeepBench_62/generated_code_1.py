import torch

def get_sobel_kernel2d_2nd_order(device=None, dtype=torch.float32):
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.t()
    gxy = get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return torch.stack((gxx, gxy, gyy), dim=0)

if __name__ == "__main__":
    device = torch.device("cpu")
    kernels = get_sobel_kernel2d_2nd_order(device=device)
    print(kernels)