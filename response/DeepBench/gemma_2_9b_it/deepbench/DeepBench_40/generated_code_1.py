import torch

def get_sobel_kernel2d(device=None, dtype=torch.float32):
  kernel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=dtype, device=device)
  kernel_y = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=dtype, device=device)
  return torch.stack((kernel_x, kernel_y))

if __name__ == "__main__":
  device = torch.device("cpu")
  kernel = get_sobel_kernel2d(device=device)
  print(kernel)