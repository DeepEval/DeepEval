import torch
import torchvision.utils as vutils

def make_grid(images, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0):
    if isinstance(images, list):
        images = torch.stack(images)

    grid = vutils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, value_range=value_range, scale_each=scale_each, pad_value=pad_value)
    return grid

if __name__ == "__main__":
    # Create a sample batch of images (e.g., 16 images of 3 channels, 32x32 pixels)
    sample_images = torch.randn(16, 3, 32, 32)  # Random images

    # Create a grid of images
    grid_image = make_grid(sample_images, nrow=4, padding=2, normalize=True)

    # Print the grid image shape
    print(grid_image.shape)  # Expected shape will depend on the parameters used