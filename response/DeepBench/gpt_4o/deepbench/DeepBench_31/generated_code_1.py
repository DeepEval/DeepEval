import torch
import torchvision.utils as vutils

def make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
        
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize,
                            range=value_range, scale_each=scale_each, pad_value=pad_value)
    return grid

if __name__ == "__main__":
    # Create a sample 4D mini-batch tensor of images
    batch_size = 16
    channels = 3
    height = 32
    width = 32
    
    # Random tensor simulating a batch of images
    images = torch.rand(batch_size, channels, height, width)

    # Create a grid of images
    grid = make_grid(images, nrow=4, padding=5, normalize=True)

    # Display the result
    print(grid)