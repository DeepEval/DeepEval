import torch

def _create_identity_grid(image_size):
    """
    Generates a grid of normalized coordinates for a given image size.

    Args:
        image_size (list): List of integers representing the dimensions of the grid (height and width).

    Returns:
        PyTorch Tensor: Tensor containing the grid coordinates, ready for use in spatial transformations.
    """
    # Calculate the number of rows and columns
    n_rows, n_cols = image_size

    # Create a range of indices for the rows and columns
    row_indices = torch.arange(n_rows, dtype=torch.float32)
    col_indices = torch.arange(n_cols, dtype=torch.float32)

    # Normalize the row and column indices
    row_indices = row_indices / (n_rows - 1)
    col_indices = col_indices / (n_cols - 1)

    # Create a grid of indices
    grid = torch.stack(torch.meshgrid([row_indices, col_indices]), dim=-1)

    # Normalize the grid coordinates
    grid = grid.to(torch.float32)
    grid = grid.div(torch.tensor([[n_rows - 1, n_cols - 1]]))

    return grid

if __name__ == "__main__":
    # Create sample input values
    image_size = [2, 3]

    # Call the function
    grid = _create_identity_grid(image_size)

    # Print the results
    print(grid)