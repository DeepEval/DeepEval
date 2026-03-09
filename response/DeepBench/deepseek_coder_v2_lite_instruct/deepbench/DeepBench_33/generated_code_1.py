import torch

def _create_identity_grid(image_size):
    height, width = image_size
    y_coords = torch.linspace(0, 1, height)
    x_coords = torch.linspace(0, 1, width)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).view(height, width, 2)
    return grid

if __name__ == "__main__":
    image_size = [3, 4]
    grid = _create_identity_grid(image_size)
    print(grid)