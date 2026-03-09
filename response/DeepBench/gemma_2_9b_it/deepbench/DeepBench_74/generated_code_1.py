import torch
import torch.nn as nn
import numpy as np

def canny(input, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps):
  if input.shape[1] == 3:
    input = input.mean(dim=1, keepdim=True)
  blurred = nn.functional.gaussian_blur(input, kernel_size=kernel_size, sigma=sigma)
  grad_x = nn.functional.conv2d(blurred, torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32))
  grad_y = nn.functional.conv2d(blurred, torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32))
  grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
  grad_angle = torch.atan2(grad_y, grad_x)
  grad_angle = (grad_angle + np.pi) / (2 * np.pi) 
  non_max = nn.functional.max_pool2d(grad_magnitude, kernel_size=5, stride=1, padding=2, return_indices=False)
  thresh1 = grad_magnitude > low_threshold
  thresh2 = grad_magnitude > high_threshold
  edges = torch.zeros_like(input)
  edges[thresh1] = 1
  if hysteresis:
    for i in range(1, edges.shape[2]):
      for j in range(1, edges.shape[3]):
        if edges[0,0,i,j] == 1:
          if (grad_magnitude[0,0,i,j] > low_threshold) and (grad_magnitude[0,0,i-1,j] > low_threshold or grad_magnitude[0,0,i+1,j] > low_threshold or grad_magnitude[0,0,i,j-1] > low_threshold or grad_magnitude[0,0,i,j+1] > low_threshold):
            edges[0,0,i,j] = 1
  return edges, edges

if __name__ == "__main__":
  input_tensor = torch.randn(1, 3, 256, 256)
  low_threshold = 0.1
  high_threshold = 0.2
  kernel_size = 5
  sigma = 1.5
  hysteresis = True
  eps = 1e-10
  
  edges, filtered_edges = canny(input_tensor, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps)
  print(f"Edges shape: {edges.shape}")
  print(f"Filtered edges shape: {filtered_edges.shape}")