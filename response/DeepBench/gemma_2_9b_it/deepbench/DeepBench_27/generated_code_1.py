from PIL import Image
import torch
import numpy as np

def to_pil_image(pic, mode=None):
  if isinstance(pic, torch.Tensor):
    pic = pic.detach().numpy()
  if not isinstance(pic, np.ndarray):
    raise TypeError("Input must be a Tensor or numpy array.")
  if pic.ndim not in (2, 3):
    raise ValueError("Input image must be 2D or 3D.")
  if pic.ndim == 2:
    pic = np.expand_dims(pic, axis=0)
  if pic.shape[0] > 4:
    raise ValueError("Input image cannot have more than 4 channels.")
  if mode is None:
    if pic.shape[2] == 1:
      mode = "L"
    elif pic.shape[2] == 2:
      mode = "LA"
    elif pic.shape[2] == 3:
      mode = "RGB"
    elif pic.shape[2] == 4:
      mode = "RGBA"
  supported_modes = {
      1: ["L", "I", "I;16", "F"],
      2: ["LA"],
      3: ["RGB", "YCbCr", "HSV"],
      4: ["RGBA", "CMYK", "RGBX"]
  }
  if mode not in supported_modes[pic.shape[2]]:
    raise ValueError(f"Mode {mode} is not supported for {pic.shape[2]} channels.")
  return Image.fromarray(pic.astype(np.uint8), mode=mode)


if __name__ == "__main__":
  # Sample input
  tensor_input = torch.randn(3, 224, 224) 
  numpy_input = np.random.rand(1, 28, 28)
  
  # Convert tensor input
  pil_image_tensor = to_pil_image(tensor_input, mode="RGB")
  print(f"PIL Image from Tensor:\n{pil_image_tensor.size}")

  # Convert numpy input
  pil_image_numpy = to_pil_image(numpy_input, mode="L")
  print(f"PIL Image from NumPy:\n{pil_image_numpy.size}")