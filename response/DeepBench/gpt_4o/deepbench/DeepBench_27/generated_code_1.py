import numpy as np
from PIL import Image
import torch

def to_pil_image(pic, mode=None):
    if not isinstance(pic, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")
    
    if isinstance(pic, torch.Tensor):
        pic = pic.cpu().numpy()
    
    if pic.ndim not in {2, 3}:
        raise ValueError("Input image must be 2D or 3D.")
    
    if pic.ndim == 3 and pic.shape[2] > 4:
        raise ValueError("Input image can have at most 4 channels.")
    
    if mode is not None:
        if pic.ndim == 2:
            if mode not in {"L", "I", "I;16", "F"}:
                raise TypeError(f"Unsupported mode '{mode}' for 1-channel image.")
        elif pic.ndim == 3:
            if pic.shape[2] == 1 and mode not in {"L", "I", "I;16", "F"}:
                raise TypeError(f"Unsupported mode '{mode}' for 1-channel image.")
            elif pic.shape[2] == 2 and mode != "LA":
                raise TypeError(f"Unsupported mode '{mode}' for 2-channel image.")
            elif pic.shape[2] == 3 and mode not in {"RGB", "YCbCr", "HSV"}:
                raise TypeError(f"Unsupported mode '{mode}' for 3-channel image.")
            elif pic.shape[2] == 4 and mode not in {"RGBA", "CMYK", "RGBX"}:
                raise TypeError(f"Unsupported mode '{mode}' for 4-channel image.")
    
    if pic.ndim == 2:
        mode = mode or "L"
    else:
        num_channels = pic.shape[2]
        if num_channels == 1:
            mode = mode or "L"
            pic = pic.squeeze(-1)
        elif num_channels == 2:
            mode = mode or "LA"
        elif num_channels == 3:
            mode = mode or "RGB"
        elif num_channels == 4:
            mode = mode or "RGBA"
    
    return Image.fromarray(pic, mode)

if __name__ == "__main__":
    # Example 1: Convert a 3-channel numpy array to a PIL Image
    array_3d = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_img = to_pil_image(array_3d)
    print("Example 1: 3-channel numpy array converted to PIL Image.")

    # Example 2: Convert a 1-channel torch tensor to a PIL Image
    tensor_2d = torch.randint(0, 255, (100, 100), dtype=torch.uint8)
    pil_img = to_pil_image(tensor_2d, mode="L")
    print("Example 2: 1-channel torch tensor converted to PIL Image.")