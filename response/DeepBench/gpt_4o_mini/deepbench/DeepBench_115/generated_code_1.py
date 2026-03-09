import numpy as np
import torch
from PIL import Image

def to_pil_image(pic, mode=None):
    if isinstance(pic, torch.Tensor):
        if pic.ndim == 2:
            pic = pic.unsqueeze(0)
        if pic.ndim == 3:
            if pic.size(0) == 1:
                pic = pic.expand(3, -1, -1)
            pic = pic.permute(1, 2, 0).numpy()
        elif pic.ndim != 4:
            raise ValueError("Invalid tensor shape")
    elif isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            pic = np.expand_dims(pic, axis=-1)
        elif pic.ndim == 3 and pic.shape[2] == 1:
            pic = np.concatenate([pic, pic, pic], axis=2)
    else:
        raise TypeError("pic should be a Tensor or numpy.ndarray")
    
    if mode is None:
        mode = 'RGB' if pic.shape[2] == 3 else 'L'
    
    return Image.fromarray(np.uint8(pic), mode)

if __name__ == "__main__":
    # Create a sample numpy array (3x100x100 RGB image)
    sample_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_image = to_pil_image(sample_array)
    pil_image.show()  # This will open the image using the default image viewer