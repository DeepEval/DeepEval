import torch

def _create_identity_grid(size):
    height, width = size
    # Create normalized coordinate grid
    y_coords = torch.linspace(-1, 1, steps=height)
    x_coords = torch.linspace(-1, 1, steps=width)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    # Stack to create a grid of shape (height, width, 2)
    grid = torch.stack((x_grid, y_grid), dim=-1)
    return grid

if __name__ == "__main__":
    size = [4, 5]  # Example size (height=4, width=5)
    grid = _create_identity_grid(size)
    print(grid)