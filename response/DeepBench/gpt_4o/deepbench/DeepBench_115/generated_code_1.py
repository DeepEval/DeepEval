import numpy as np
from PIL import Image
import torch

def to_pil_image(pic, mode=None):
    if isinstance(pic, torch.Tensor):
        if pic.ndimension() == 3:
            pic = pic.permute(1, 2, 0)  # Change from CxHxW to HxWxC
        elif pic.ndimension() == 2:
            pass
        else:
            raise ValueError(f"Unsupported tensor dimension: {pic.ndimension()}")
        pic = pic.numpy()

    if isinstance(pic, np.ndarray):
        if pic.ndim == 3 and pic.shape[2] == 1:  # Single channel image
            pic = pic[:, :, 0]
        elif pic.ndim == 2:
            pass
        elif pic.ndim == 3 and pic.shape[2] in [3, 4]:  # RGB or RGBA
            pass
        else:
            raise ValueError(f"Unsupported ndarray dimension or shape: {pic.shape}")
    else:
        raise TypeError(f"Unsupported input type: {type(pic)}")

    if mode is None:
        if pic.ndim == 2:
            mode = "L"
        elif pic.ndim == 3 and pic.shape[2] == 3:
            mode = "RGB"
        elif pic.ndim == 3 and pic.shape[2] == 4:
            mode = "RGBA"

    return Image.fromarray(pic, mode=mode)

if __name__ == "__main__":
    # Create a sample torch Tensor and convert it to a PIL Image
    tensor_example = torch.rand(3, 100, 100)  # Random RGB image
    pil_image_from_tensor = to_pil_image(tensor_example)
    print(f"Converted tensor to PIL image: {pil_image_from_tensor}")

    # Create a sample numpy array and convert it to a PIL Image
    ndarray_example = np.random.rand(100, 100, 3) * 255  # Random RGB image
    pil_image_from_ndarray = to_pil_image(ndarray_example.astype(np.uint8))
    print(f"Converted ndarray to PIL image: {pil_image_from_ndarray}")