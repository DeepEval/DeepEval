import torch

def _create_identity_grid(grid_size):
  height, width = grid_size
  grid_x = torch.linspace(-1, 1, width)
  grid_y = torch.linspace(-1, 1, height)
  grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
  return torch.stack((grid_x, grid_y), dim=-1)

if __name__ == "__main__":
  grid_size = [3, 4]
  grid = _create_identity_grid(grid_size)
  print(grid)