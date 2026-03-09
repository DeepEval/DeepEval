import torch

def make_grid(images, nrow=8, padding=2, normalize=True, value_range=(0, 1), scale_each=False, pad_value=0):
  if isinstance(images, torch.Tensor):
    images = [images]
  if normalize:
    images = [(x - value_range[0]) / (value_range[1] - value_range[0]) for x in images]
  if scale_each:
    images = [x / torch.max(x) for x in images]
  grid = torch.cat([torch.cat(images[i:i + nrow], dim=1) for i in range(0, len(images), nrow)], dim=0)
  if padding > 0:
    grid = torch.nn.functional.pad(grid, (0, padding, 0, padding), value=pad_value)
  return grid

if __name__ == "__main__":
  images = [torch.rand(1, 3, 32, 32) for _ in range(16)]
  grid = make_grid(images, nrow=4)
  print(grid.shape)