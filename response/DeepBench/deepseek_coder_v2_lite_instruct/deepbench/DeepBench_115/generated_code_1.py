import torch
from PIL import Image
import numpy as np

def to_pil_image(pic, mode=None):
    if not isinstance(pic, (torch.Tensor, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}'.format(type(pic)))
    
    if isinstance(pic, torch.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))
        if pic.ndimension() == 2:
            pic = pic.unsqueeze(0)

        if mode is not None and mode != pic.mode:
            pic = pic.convert(mode)

        return Image.fromarray(pic.numpy().astype(np.uint8))
    
    if isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))
        if pic.ndim == 2:
            pic = np.expand_dims(pic, 0)

        if mode is not None and mode != pic.shape[2] * (pic.ndim - 1):
            raise ValueError("Unsupported mode '{}' for image array conversion.".format(mode))

        return Image.fromarray(pic.astype(np.uint8))

if __name__ == "__main__":
    # Example usage
    tensor_example = torch.rand(3, 256, 256)  # Example tensor with shape (3, 256, 256)
    array_example = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)  # Example numpy array with shape (256, 256, 3)

    pil_tensor_image = to_pil_image(tensor_example, mode='RGB')
    pil_array_image = to_pil_image(array_example, mode='RGB')

    print("Tensor to PIL Image:", pil_tensor_image)
    print("Array to PIL Image:", pil_array_image)