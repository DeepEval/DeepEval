import torch

def xavier_normal_(tensor, gain=1.0, generator=None):
  if generator is None:
    fan_in = tensor.size(1)
    fan_out = tensor.size(0)
    std = gain / torch.sqrt(fan_in + fan_out)
    return tensor.normal_(0, std)
  else:
    fan_in = tensor.size(1)
    fan_out = tensor.size(0)
    std = gain / torch.sqrt(fan_in + fan_out)
    return tensor.normal_(0, std, generator=generator)

if __name__ == "__main__":
  tensor = torch.empty(2, 3)
  print(f"Input Tensor:\n{tensor}")
  xavier_normal_(tensor, gain=2.0)
  print(f"Tensor after Xavier Normal Initialization:\n{tensor}")