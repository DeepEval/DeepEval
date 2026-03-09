import torch

def get_sobel_kernel_5x5_2nd_order_xy(k_size=5):
    middle = k_size // 2
    g_xx = torch.tensor([
        [-0.0375, -0.0750, -0.0750, -0.0750, -0.0375],
        [ 0.0750, -0.1500, -0.3000, -0.1500,  0.0750],
        [ 0.0750, -0.3000, -0.6000, -0.3000,  0.0750],
        [ 0.0750, -0.1500, -0.3000, -0.1500,  0.0750],
        [ 0.0375,  0.0750,  0.0750,  0.0750,  0.0375]
    ], device=args.device, dtype=args.dtype)

    g_yy = torch.flip(g_xx, [0])

    g_xy = torch.tensor([
        [0.0375, 0.0750, 0.0750, 0.0750, 0.0375],
        [0.0750, 0.1500, 0.3000, 0.1500, 0.0750],
        [0.0750, 0.3000, 0.6000, 0.3000, 0.0750],
        [0.0750, 0.1500, 0.3000, 0.1500, 0.0750],
        [0.0375, 0.0750, 0.0750, 0.0750, 0.0375]
    ], device=args.device, dtype=args.dtype)

    return torch.stack([g_xx, g_xy, g_yy])

def get_sobel_kernel_5x5_2nd_order(device=None, dtype=None):
    gxx = get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype).T
    gyy = get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)

    gxy = get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)

    return torch.stack([gxx, gxy, gyy])

def get_sobel_kernel2d_2nd_order(device=None, dtype=None):
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype).T

    gxy = get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)

    return torch.stack([gxx, gxy, gyy])

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    kernel = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    print(kernel)