import torch
from PIL import Image
import numpy as np

def to_pil_image(pic, mode='RGB'):
    if isinstance(pic, torch.Tensor):
        # Convert tensor to numpy array
        pic_np = pic.numpy()
    elif isinstance(pic, np.ndarray):
        pic_np = pic
    else:
        raise TypeError("Input must be a Tensor or numpy array.")

    if pic_np.ndim > 3:
        raise ValueError("Images with more than 4 channels are not supported.")

    # Convert numpy array to PIL Image
    if len(pic_np.shape) == 3:  # 3D array (RGB)
        if mode == 'LA' or mode == 'RGB' or mode == 'YCbCr' or mode == 'HSV':
            output_img = Image.fromarray(pic_np)
        elif mode == 'RGBA':
            output_img = Image.fromarray(pic_np, mode='RGBA')
        elif mode == 'RGBX':
            output_img = Image.fromarray(pic_np, mode='RGBX')
        else:
            raise ValueError(f"Mode {mode} is not compatible with 3-channel input.")
    elif len(pic_np.shape) == 2:  # 2D array (grayscale)
        if mode == 'L':
            output_img = Image.fromarray(pic_np)
        elif mode == 'I':
            output_img = Image.fromarray(pic_np, mode='L')
        elif mode == 'I;16':
            output_img = Image.fromarray(pic_np, mode='I;16')
        elif mode == 'F':
            output_img = Image.fromarray(pic_np)
        else:
            raise ValueError(f"Mode {mode} is not compatible with 1-channel input.")
    else:
        raise ValueError("Input must be a 2D or 3D image.")

    return output_img

if __name__ == "__main__":
    # Sample input tensor
    tensor = torch.randn(3, 256, 256)  # 3x256x256 grayscale tensor

    # Convert tensor to PIL Image
    pil_image = to_pil_image(tensor)
    pil_image.save("output_image.png")