import torch

def get_sobel_kernel2d_2nd_order(device=torch.device("cpu"), dtype=torch.float32):
    gxx = torch.Tensor([
        [0.003054, 0.013281, 0.021969, 0.013281, 0.003054],
        [0.013281, 0.059692, 0.098438, 0.059692, 0.013281],
        [0.021969, 0.098438, 0.162109, 0.098438, 0.021969],
        [0.013281, 0.059692, 0.098438, 0.059692, 0.013281],
        [0.003054, 0.013281, 0.021969, 0.013281, 0.003054],
    ], device=device, dtype=dtype)
    gyy = gxx.t()
    gxy = torch.Tensor([
        [-0.546296, -0.812500, -0.546296],
        [-0.812500, 4.604375, 4.604375],
        [-0.546296, -0.812500, -0.546296],
    ], device=device, dtype=dtype)
    kernels = torch.stack((gxx, gxy, gyy), dim=0)
    return kernels

if __name__ == "__main__":
    kernels = get_sobel_kernel2d_2nd_order(device=torch.device("cpu"), dtype=torch.float32)
    print(kernels)