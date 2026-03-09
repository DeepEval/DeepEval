import torch
import torch.nn as nn

def deform_conv2d(input, offset, weight, bias=None, stride=1, padding=0, dilation=1, mask=None):
  if mask is not None:
    return nn.DeformableConv2d(input.shape[1], weight.shape[0], kernel_size=weight.shape[2], stride=stride, padding=padding, dilation=dilation)(input, offset, mask)
  else:
    return nn.Conv2d(input.shape[1], weight.shape[0], kernel_size=weight.shape[2], stride=stride, padding=padding, dilation=dilation)(input, weight, bias)

if __name__ == "__main__":
  input_tensor = torch.randn(1, 3, 16, 16)
  offset_tensor = torch.randn(1, 2, 8, 8)
  weight_tensor = torch.randn(32, 3, 3, 3)
  
  result = deform_conv2d(input_tensor, offset_tensor, weight_tensor)
  print(result.shape)