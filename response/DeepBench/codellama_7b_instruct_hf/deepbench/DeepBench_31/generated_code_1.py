import torch

def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Generates a grid of images.

    Arguments:
        tensor (torch.Tensor or list of torch.Tensors): the input tensor.
        nrow (int, optional): number of images per row in the grid. Default: 8.
        padding (int, optional): number of pixels between images in the grid. Default: 2.
        normalize (bool, optional): whether to normalize the images. Default: False.
        value_range (tuple, optional): minimum and maximum values for normalization. Default: None.
        scale_each (bool, optional): whether to scale each image separately. Default: False.
        pad_value (float, optional): value for padded pixels. Default: 0.

    Returns:
        torch.Tensor: a tensor containing the grid of images.
    """
    if not isinstance(tensor, list):
        tensor = [tensor]

    n_images = len(tensor)

    if n_images == 1:
        return tensor

    # Calculate the height and width of the grid
    height, width = tensor[0].shape[1], tensor[0].shape[2]

    for i in range(1, n_images):
        height = max(height, tensor[i].shape[1])
        width = max(width, tensor[i].shape[2])

    # Calculate the number of rows and columns in the grid
    n_rows = int(n_images / nrow) if n_images % nrow == 0 else int(n_images / nrow) + 1
    n_cols = n_images // n_rows

    # Create a blank grid
    grid = torch.zeros((height * n_rows + padding, width * n_cols + padding, tensor[0].shape[0]))

    # Add the images to the grid
    for i in range(n_images):
        row = int(i // n_cols)
        col = i % n_cols
        grid[row * (height + padding): (row + 1) * (height + padding), col * (width + padding): (col + 1) * (width + padding)] = tensor[i]

    return grid

if __name__ == "__main__":
    tensor1 = torch.randn(3, 32, 32)
    tensor2 = torch.randn(3, 32, 32)
    tensor3 = torch.randn(3, 32, 32)
    tensor_list = [tensor1, tensor2, tensor3]
    grid = make_grid(tensor_list, nrow=2, padding=1, normalize=True)
    print("Grid shape:", grid.shape)