import torch

def kaiming_uniform_(tensor, a=0.0, mode='fan_in', nonlinearity='relu', generator=None):
  gain = torch.nn.init.calculate_gain(nonlinearity, a)
  num_fan = tensor.size(0) if mode == 'fan_in' else tensor.size(1)
  bound = gain * torch.sqrt(2.0 / num_fan)
  return torch.nn.init.uniform_(tensor, -bound, bound, generator=generator)

if __name__ == "__main__":
  tensor = torch.empty(2, 3)
  print(f"Original Tensor: \n{tensor}")
  kaiming_uniform_(tensor, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
  print(f"Tensor after Kaiming Uniform Initialization: \n{tensor}")