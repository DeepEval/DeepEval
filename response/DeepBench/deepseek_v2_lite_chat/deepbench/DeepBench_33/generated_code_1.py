import torch

def _create_identity_grid(size):
    """
    Generates a grid of normalized coordinates for a given image size.

    Args:
    size (list of ints): A list of two integers representing the dimensions of the grid
    (height and width).

    Returns:
    torch.Tensor: A 3D tensor containing the grid coordinates with shape (H, W, 2),
    where H is the height and W is the width.
    """
    # Create a grid of indices (i, j) with shape (H, W, 2)
    y_indices = torch.arange(size[0]).view(size[0], 1)
    x_indices = torch.arange(size[1]).view(1, size[1], 1)
    grid = torch.stack([x_indices, y_indices], dim=3).float()

    # Normalize the coordinates to be in the range [0, 1]
    grid /= (size[1] - 1)
    grid /= (size[0] - 1)

    return grid

if __name__ == "__main__":
    # Example usage
    grid = _create_identity_grid([4, 4])
    print("Grid shape:", grid.shape)
    print("Grid:", grid)

    # You can use grid for spatial transformations like bilinear interpolation
    # ...