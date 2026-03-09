import torch

def _create_identity_grid(size):
    height, width = size
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).float()
    grid = (grid / torch.tensor([width - 1, height - 1])).unsqueeze(0)
    return grid

if __name__ == "__main__":
    size = [4, 5]  # Example input: height = 4, width = 5
    grid = _create_identity_grid(size)
    print(grid)