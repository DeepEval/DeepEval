import torch

def get_sobel_kernel2d(device=None, dtype=None):
    sobel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], device=device, dtype=dtype)
    
    sobel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], device=device, dtype=dtype)
    
    return torch.stack((sobel_x, sobel_y))

if __name__ == "__main__":
    kernel = get_sobel_kernel2d(device='cpu', dtype=torch.float32)
    print("Sobel Kernels:")
    print(kernel)