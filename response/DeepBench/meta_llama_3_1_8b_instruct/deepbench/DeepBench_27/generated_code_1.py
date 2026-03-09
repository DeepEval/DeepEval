import torch
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

def to_pil_image(pic, mode=None):
    if isinstance(pic, torch.Tensor):
        pic = pic.numpy()
    if isinstance(pic, np.ndarray):
        if len(pic.shape) == 2:
            if mode == "I;16":
                img = Image.frombytes("I;16", pic.shape, pic.astype(np.uint16).tobytes())
            elif mode == "L":
                img = Image.frombytes("L", pic.shape, pic.astype(np.uint8).tobytes())
            elif mode == "I":
                img = Image.frombytes("I", pic.shape, pic.astype(np.int32).tobytes())
            elif mode == "F":
                img = Image.frombytes("F", pic.shape, pic.astype(np.float32).tobytes())
            else:
                raise ValueError("Invalid mode for 2D image")
        elif len(pic.shape) == 3:
            if pic.shape[2] == 1:
                if mode == "I;16":
                    img = Image.frombytes("L;16", pic.shape, pic.astype(np.uint16).tobytes())
                elif mode == "L":
                    img = Image.frombytes("L", pic.shape, pic.astype(np.uint8).tobytes())
                elif mode == "I":
                    img = Image.frombytes("L", pic.shape, pic.astype(np.int32).tobytes())
                elif mode == "F":
                    img = Image.frombytes("F", pic.shape, pic.astype(np.float32).tobytes())
                else:
                    raise ValueError("Invalid mode for 2D image")
            elif pic.shape[2] == 2:
                if mode == "LA":
                    img = Image.frombytes("LA", pic.shape, pic.astype(np.uint8).tobytes())
                else:
                    raise ValueError("Invalid mode for 2D image")
            elif pic.shape[2] == 3:
                if mode == "RGB":
                    img = Image.frombytes("RGB", pic.shape, pic.astype(np.uint8).tobytes())
                elif mode == "YCbCr":
                    img = Image.frombytes("YCbCr", pic.shape, pic.astype(np.uint8).tobytes())
                elif mode == "HSV":
                    img = Image.frombytes("HSV", pic.shape, pic.astype(np.uint8).tobytes())
                else:
                    raise ValueError("Invalid mode for 2D image")
            elif pic.shape[2] == 4:
                if mode == "RGBA":
                    img = Image.frombytes("RGBA", pic.shape, pic.astype(np.uint8).tobytes())
                elif mode == "CMYK":
                    img = Image.frombytes("CMYK", pic.shape, pic.astype(np.uint8).tobytes())
                elif mode == "RGBX":
                    img = Image.frombytes("RGBX", pic.shape, pic.astype(np.uint8).tobytes())
                else:
                    raise ValueError("Invalid mode for 2D image")
            else:
                raise ValueError("Image has more than 4 channels")
        else:
            raise ValueError("Image is not 2D or 3D")
    else:
        raise TypeError("Input is not a Tensor or numpy array")
    return img

if __name__ == "__main__":
    # Create sample input values
    pic1 = torch.ones(3, 3, 3) * 255
    pic2 = np.ones((3, 3, 1)) * 255
    pic3 = np.ones((3, 3, 4)) * 255

    # Call the function and print the results
    img1 = to_pil_image(pic1)
    img2 = to_pil_image(pic2, mode="L")
    img3 = to_pil_image(pic3, mode="RGBA")

    print(img1.mode)
    print(img2.mode)
    print(img3.mode)

    img1.show()
    img2.show()
    img3.show()