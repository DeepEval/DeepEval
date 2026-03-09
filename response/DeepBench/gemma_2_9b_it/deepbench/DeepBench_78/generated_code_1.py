import torch
import torch.nn.functional as F

def filter2d(input, kernel, border_type, normalized, padding, behaviour='corr'):
  if normalized:
    kernel = kernel / torch.sum(kernel)
  if behaviour == 'conv':
    kernel = torch.flip(kernel, [2,3])
  if padding == 'same':
    padding_h = (kernel.shape[1]-1)//2
    padding_w = (kernel.shape[2]-1)//2
    input = F.pad(input, (0,0,padding_h,padding_h,padding_w,padding_w), mode=border_type)
  elif padding == 'valid':
    pass
  else:
    raise ValueError(f"Invalid padding mode: {padding}. Choose from 'same' or 'valid'.")
  return F.conv2d(input, kernel, padding=0, groups=input.shape[1], bias=False)

if __name__ == "__main__":
  input_tensor = torch.randn(1, 3, 10, 10)
  kernel = torch.randn(1, 3, 3)
  output_tensor = filter2d(input_tensor, kernel, border_type='constant', normalized=False, padding='same', behaviour='corr')
  print(output_tensor.shape)