import torch
from PIL import Image
import numpy as np

def to_pil_image(pic, mode=None):
  if isinstance(pic, torch.Tensor):
    if pic.ndim == 3:
      pic = pic.squeeze()
    pic = pic.cpu().numpy()
  elif not isinstance(pic, np.ndarray):
    raise TypeError("Input must be a Tensor or a numpy.ndarray")
  if len(pic.shape) == 2:
    pic = np.stack((pic, pic, pic), axis=2)
  if mode is None:
    mode = 'RGB' if pic.shape[2] == 3 else 'L'
  return Image.fromarray(pic.astype(np.uint8), mode)

if __name__ == "__main__":
  tensor_input = torch.randn(1, 3, 224, 224)
  pil_image_tensor = to_pil_image(tensor_input)
  print(f'PIL Image from Tensor: {pil_image_tensor}')

  array_input = np.random.rand(224, 224, 3)
  pil_image_array = to_pil_image(array_input)
  print(f'PIL Image from Array: {pil_image_array}')