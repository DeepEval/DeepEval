import torch

def _create_identity_grid(size):
    """Generates a grid of normalized coordinates for a given image size.

    Args:
        size: A list of integers representing the dimensions of the grid (height and width).

    Returns:
        A PyTorch Tensor containing the grid coordinates, ready for use in spatial transformations.
    """
    h, w = size

    # Create a meshgrid of normalized coordinates
    y, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))

    # Reshape the meshgrid to a 2D tensor
    grid = torch.stack([x, y], dim=-1)

    return grid

if __name__ == "__main__":
    # Example usage
    size = [32, 32]
    grid = _create_identity_grid(size)

    # Print the grid coordinates
    print(grid)