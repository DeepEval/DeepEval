import numpy as np
import torch
from PIL import Image

def to_pil_image(pic, mode=None):
    if not isinstance(pic, (torch.Tensor, np.ndarray)):
        raise TypeError("Input must be a Tensor or numpy.ndarray.")

    if isinstance(pic, torch.Tensor):
        pic = pic.detach().cpu().numpy()

    if pic.ndim not in {2, 3}:
        raise ValueError("Input image must be 2D or 3D.")
    
    if pic.ndim == 2:
        channels = 1
    else:
        channels = pic.shape[2]
    
    if channels > 4:
        raise ValueError("Input image cannot have more than 4 channels.")

    if mode is None:
        if channels == 1:
            mode = 'L'
        elif channels == 2:
            mode = 'LA'
        elif channels == 3:
            mode = 'RGB'
        elif channels == 4:
            mode = 'RGBA'
    
    valid_modes = {
        1: ['L', 'I', 'I;16', 'F'],
        2: ['LA'],
        3: ['RGB', 'YCbCr', 'HSV'],
        4: ['RGBA', 'CMYK', 'RGBX']
    }

    if mode not in valid_modes.get(channels, []):
        raise ValueError(f"Incompatible mode '{mode}' for input with {channels} channels.")
    
    return Image.fromarray(pic.astype(np.uint8), mode)

if __name__ == "__main__":
    # Create a sample grayscale image (1-channel)
    sample_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    pil_image = to_pil_image(sample_array)
    pil_image.show()

    # Create a sample RGB image (3-channel)
    sample_rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    pil_image_rgb = to_pil_image(sample_rgb_array)
    pil_image_rgb.show()