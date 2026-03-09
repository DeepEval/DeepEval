import torch
from torchvision.utils import make_grid

def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=(-1, 1), scale_each=False, pad_value=0):
    """
    Generate a grid of images from a 4D mini-batch tensor.
    
    Parameters:
    tensor (torch.Tensor or list of torch.Tensors): 4D mini-batch of images.
    nrow (int): Number of images per row in the grid.
    padding (int): Padding around each image in the grid.
    normalize (bool): Whether to shift images to the range 0-1.
    value_range (tuple of 2 ints): Min and max values for normalization.
    scale_each (bool): Whether to scale each image separately.
    pad_value (int): Value for padded pixels in the grid.
    
    Returns:
    torch.Tensor: The grid of images.
    """
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, 0)  # Convert list to stacked tensor
    
    # Convert the tensor to a grid of images
    grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range, scale_each=scale_each)
    
    # Convert the tensor image into a numpy image
    grid = grid.permute(1, 2, 0).clamp(0, 1).to('cpu').numpy()
    
    # Handle padding
    if padding > 0:
        grid = grid[:-padding, :-padding, :]
    
    # Handle scale_each
    if scale_each:
        grid = grid / (2 ** padding)
    
    # Add the pad_value to the grid if it's a torch tensor
    if isinstance(pad_value, torch.Tensor):
        grid = grid + pad_value.to('cpu').unsqueeze(0).unsqueeze(0)
    
    return torch.from_numpy(grid)

if __name__ == "__main__":
    import numpy as np

    # Example tensor
    batch_size = 10
    height = 32
    width = 32
    channels = 3
    tensor = torch.rand(batch_size, channels, height, width)
    
    # Create a grid
    nrow = 2
    padding = 1
    normalize = True
    value_range = (-1, 1)
    scale_each = True
    pad_value = torch.tensor([0.5, 0.5, 0.5])
    
    grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range, scale_each=scale_each, pad_value=pad_value)
    
    print("Grid shape:", grid.shape)
    print("Grid min:", grid.min())
    print("Grid max:", grid.max())