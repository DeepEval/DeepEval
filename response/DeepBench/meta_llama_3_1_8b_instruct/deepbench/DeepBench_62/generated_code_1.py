import torch
import torch.nn as nn
import numpy as np

def get_sobel_kernel_5x5_2nd_order(direction):
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size))
    if direction == 'x':
        kernel[1, 1] = -1
        kernel[1, 0] = -1
        kernel[1, 2] = -1
        kernel[1, 3] = -1
        kernel[0, 1] = -1
        kernel[2, 1] = -1
        kernel[3, 1] = -1
        kernel[4, 1] = -1
        kernel[2, 0] = 8
        kernel[2, 2] = 8
        kernel[2, 3] = 8
        kernel[2, 4] = -1
        kernel[0, 0] = -1
        kernel[0, 2] = -1
        kernel[0, 3] = -1
        kernel[0, 4] = -1
        kernel[4, 0] = -1
        kernel[4, 2] = -1
        kernel[4, 3] = -1
        kernel[4, 4] = -1
    elif direction == 'y':
        kernel[1, 1] = -1
        kernel[0, 1] = -1
        kernel[2, 1] = -1
        kernel[3, 1] = -1
        kernel[4, 1] = -1
        kernel[1, 0] = -1
        kernel[1, 2] = -1
        kernel[1, 3] = -1
        kernel[1, 4] = -1
        kernel[2, 2] = 8
        kernel[2, 0] = 8
        kernel[2, 3] = 8
        kernel[2, 4] = 8
        kernel[0, 0] = -1
        kernel[0, 2] = -1
        kernel[0, 3] = -1
        kernel[0, 4] = -1
        kernel[3, 0] = -1
        kernel[4, 0] = -1
        kernel[3, 2] = -1
        kernel[4, 2] = -1
        kernel[3, 4] = -1
        kernel[4, 4] = -1
    else:
        raise ValueError("Invalid direction. Direction should be either 'x' or 'y'.")
    
    return torch.tensor(kernel, dtype=torch.float32)

def _get_sobel_kernel_5x5_2nd_order_xy():
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[0, 1] = -1
    kernel[0, 2] = -1
    kernel[0, 3] = -1
    kernel[1, 2] = -1
    kernel[2, 2] = 8
    kernel[3, 2] = -1
    kernel[4, 2] = -1
    kernel[1, 0] = -1
    kernel[2, 0] = 8
    kernel[3, 0] = -1
    kernel[4, 0] = -1
    kernel[0, 0] = -1
    kernel[0, 4] = -1
    kernel[1, 4] = -1
    kernel[3, 4] = -1
    kernel[4, 4] = -1
    kernel[2, 4] = -1
    return torch.tensor(kernel, dtype=torch.float32)

def get_sobel_kernel2d_2nd_order(device=None, dtype=torch.float32):
    gxx = get_sobel_kernel_5x5_2nd_order('x')
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack((gxx, gxy, gyy), dim=0).to(device=device, dtype=dtype)

if __name__ == "__main__":
    kernel = get_sobel_kernel2d_2nd_order(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(kernel.shape)
    print(kernel)