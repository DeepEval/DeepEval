import torch
import torchvision.utils as utils

def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors are expected, got {}'.format(type(tensor)))

    if not isinstance(nrow, int):
        raise TypeError('Expected int for nrow, got {}'.format(type(nrow)))

    if nrow <= 0:
        raise ValueError('Expected nrow > 0, got {}'.format(nrow))

    if not isinstance(padding, int):
        raise TypeError('Expected int for padding, got {}'.format(type(padding)))

    if padding < 0:
        raise ValueError('Expected padding >= 0, got {}'.format(padding))

    if normalize:
        if value_range is None:
            raise ValueError('Must specify value_range when normalize is True')
        if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
            raise ValueError('value_range must be a list or tuple with two elements')
        if not all(isinstance(i, (int, float)) for i in value_range):
            raise TypeError('All elements in value_range must be int or float')

    if scale_each:
        if not isinstance(scale_each, bool):
            raise TypeError('Expected bool for scale_each, got {}'.format(type(scale_each)))

    if pad_value is not None and not isinstance(pad_value, (int, float)):
        raise TypeError('Expected int or float for pad_value, got {}'.format(type(pad_value)))

    if isinstance(tensor, list):
        tensor = torch.stack(tensor)

    return utils.make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range, scale_each=scale_each, pad_value=pad_value)

if __name__ == "__main__":
    # Create a batch of 16 images (4x4)
    batch_of_images = [torch.rand(3, 64, 64) for _ in range(16)]
    
    # Generate the grid
    grid = make_grid(batch_of_images, nrow=4, padding=2, normalize=True, value_range=(0, 1), scale_each=False, pad_value=0)
    
    # Print the grid
    print(grid)