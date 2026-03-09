import numpy as np
from PIL import Image

def to_pil_image(pic, mode=None):
    if not isinstance(pic, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a Tensor or numpy.ndarray")

    if not isinstance(pic.shape, (tuple, list)):
        raise ValueError("Input must be 2D or 3D")

    if len(pic.shape) == 2:
        if mode is None:
            mode = "L"
        elif mode not in ["L", "I", "I;16", "F"]:
            raise ValueError("Unsupported mode for 2D input")

    elif len(pic.shape) == 3:
        if mode is None:
            mode = "RGB"
        elif mode not in ["RGB", "YCbCr", "HSV", "RGBX", "RGBA", "CMYK"]:
            raise ValueError("Unsupported mode for 3D input")

    else:
        raise ValueError("Input must be 2D or 3D")

    if pic.shape[2] > 4:
        raise ValueError("Input must have at most 4 channels")

    pil_image = Image.fromarray(pic)
    pil_image.convert(mode)

    return pil_image

if __name__ == "__main__":
    import torch
    img_np = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    pil_img = to_pil_image(img_np, mode="RGB")
    print("PIL image mode:", pil_img.mode)
    print("PIL image size:", pil_img.size)
    img_tensor = torch.randint(0, 255, size=(256, 256), dtype=torch.uint8).numpy()
    pil_img_2d = to_pil_image(img_tensor, mode="L")
    print("2D PIL image mode:", pil_img_2d.mode)
    print("2D PIL image size:", pil_img_2d.size)