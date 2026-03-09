import torch
import torchvision.utils

def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Creates a grid of images from a tensor.

    Args:
        tensor: A tensor (either a torch.Tensor or a list of torch.Tensors) representing a 4D mini-batch of images.
        nrow: Number of images per row in the grid.
        padding: Padding between images in the grid.
        normalize: Whether to shift images to the range 0-1.
        value_range: Min and max values for normalization.
        scale_each: Whether to scale each image separately.
        pad_value: Value for padded pixels.

    Returns:
        A torch.Tensor containing the grid of images.
    """

    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if normalize:
        if value_range is not None:
            min_value, max_value = value_range
        else:
            min_value, max_value = tensor.min(), tensor.max()

        tensor = (tensor - min_value) / (max_value - min_value)

    return torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value)

if __name__ == "__main__":
    # Create sample input values
    sample_input = torch.randn(64, 3, 28, 28)

    # Call the function and print the results
    grid = make_grid(sample_input)
    print(grid)