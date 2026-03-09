import torch
from PIL import Image
import numpy as np

def to_pil_image(pic, mode=None):
    if not isinstance(pic, (torch.Tensor, np.ndarray)):
        raise TypeError("Input should be a Tensor or numpy array.")
    
    if isinstance(pic, torch.Tensor):
        if not pic.ndim == 2 and not pic.ndim == 3:
            raise ValueError("Input tensor should be 2D or 3D.")
        if pic.ndim == 2:
            pic = pic.unsqueeze(2)
        if mode is not None:
            if mode not in Image.MODES:
                raise ValueError(f"Unsupported mode: {mode}")
            if mode not in Image.getmodebasemaps().values():
                raise TypeError(f"Mode {mode} is not supported for tensor input.")
        if pic.shape[0] == 1:
            pic = torch.cat([pic, pic, pic])
        elif pic.shape[0] == 2:
            pic = torch.cat([pic, torch.zeros(1, pic.shape[1], pic.shape[2])])
        elif pic.shape[0] == 4:
            pic = pic[:3]
        return Image.fromarray(pic.permute(1, 2, 0).numpy().astype(np.uint8), mode=mode)
    
    if isinstance(pic, np.ndarray):
        if not pic.ndim == 2 and not pic.ndim == 3:
            raise ValueError("Input numpy array should be 2D or 3D.")
        if pic.ndim == 2:
            pic = np.expand_dims(pic, 2)
        if mode is not None:
            if mode not in Image.MODES:
                raise ValueError(f"Unsupported mode: {mode}")
            if mode not in Image.getmodebasemaps().values():
                raise TypeError(f"Mode {mode} is not supported for numpy array input.")
        if pic.shape[0] == 1:
            pic = np.concatenate([pic, pic, pic], axis=0)
        elif pic.shape[0] == 2:
            pic = np.concatenate([pic, np.zeros((1,) + pic.shape[1:])], axis=0)
        elif pic.shape[0] == 4:
            pic = pic[:3]
        return Image.fromarray(pic.transpose(1, 2, 0).astype(np.uint8), mode=mode)

    return output

if __name__ == "__main__":
    # Test with a 3D tensor (RGB image)
    tensor_rgb = torch.rand(3, 256, 256)
    pil_rgb = to_pil_image(tensor_rgb, mode="RGB")
    print("RGB PIL Image:", pil_rgb)

    # Test with a 2D tensor (grayscale image)
    tensor_gray = torch.rand(1, 256, 256)
    pil_gray = to_pil_image(tensor_gray, mode="L")
    print("Grayscale PIL Image:", pil_gray)

    # Test with a numpy array (RGB image)
    np_array_rgb = np.random.rand(256, 256, 3)
    pil_rgb_np = to_pil_image(np_array_rgb, mode="RGB")
    print("RGB PIL Image from numpy array:", pil_rgb_np)

    # Test with a numpy array (grayscale image)
    np_array_gray = np.random.rand(256, 256)
    pil_gray_np = to_pil_image(np_array_gray, mode="L")
    print("Grayscale PIL Image from numpy array:", pil_gray_np)