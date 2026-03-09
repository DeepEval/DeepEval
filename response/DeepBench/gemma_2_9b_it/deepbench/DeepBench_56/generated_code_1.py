import torch
from PIL import Image

def encode_jpeg(images, quality=75):
  if not 1 <= quality <= 100:
    raise ValueError("Quality must be between 1 and 100")
  if not images:
    raise ValueError("Input list cannot be empty")
  if isinstance(images, torch.Tensor):
    images = [images]
  outputs = []
  for image in images:
    image = image.squeeze(0).detach().numpy()
    image = Image.fromarray(image.astype(np.uint8))
    with io.BytesIO() as output:
      image.save(output, format="JPEG", quality=quality)
      contents = output.getvalue()
    outputs.append(contents)
  return outputs if len(images) > 1 else outputs[0]

if __name__ == "__main__":
  sample_image = torch.randn(1, 3, 224, 224)
  encoded_image = encode_jpeg(sample_image)
  print(f"Encoded image size: {len(encoded_image)} bytes")