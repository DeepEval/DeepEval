import torch
import numpy as np
from PIL import Image

def to_pil_image(pic, mode=None):
    # Check if input is a tensor or ndarray
    if isinstance(pic, torch.Tensor):
        # Convert tensor to numpy array
        pic = pic.detach().cpu().numpy()
    elif not isinstance(pic, np.ndarray):
        raise ValueError('pic should be Tensor or ndarray')

    # Check if mode is specified
    if mode is not None:
        raise NotImplementedError('Mode is not supported for Tensor or ndarray input.')

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(pic)

    return pil_image

if __name__ == "__main__":
    # Create sample input values
    tensor = torch.randn(3, 4, 5)
    ndarray = np.random.rand(3, 4, 5)

    # Call function and print results
    pil_image1 = to_pil_image(tensor)
    pil_image2 = to_pil_image(ndarray)

    print(f'PIL Image created from tensor: {pil_image1}')
    print(f'PIL Image created from ndarray: {pil_image2}')