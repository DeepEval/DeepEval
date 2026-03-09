import torch

def get_sobel_kernel2d(device=None, dtype=torch.float32):
    kernel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    kernel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    kernel_x = torch.tensor(kernel_x, device=device, dtype=dtype)
    kernel_y = torch.tensor(kernel_y, device=device, dtype=dtype)
    return torch.stack((kernel_x, kernel_y), dim=0)

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    sobel_kernel = get_sobel_kernel2d(device, dtype)
    print(sobel_kernel)