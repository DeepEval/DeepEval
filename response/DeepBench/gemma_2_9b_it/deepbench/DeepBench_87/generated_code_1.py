import torch
from torchvision.transforms import Normalize

def flow_to_image(flow):
  if flow.shape not in [(2, *_), (2, 2, *_) ] or flow.dtype != torch.float:
    raise ValueError("Input flow tensor must have shape (2, H, W) or (N, 2, H, W) and dtype torch.float.")
  if flow.shape[0] == 2:
    flow = flow.unsqueeze(0)
  flow_norm = Normalize((0.5,), (0.5,))
  flow_rgb = flow_norm(flow.permute(0, 2, 3, 1)) * 255
  return flow_rgb.permute(0, 3, 1, 2).type(torch.uint8) 

if __name__ == "__main__":
  flow = torch.randn(2, 128, 128)
  image = flow_to_image(flow)
  print(image.shape)