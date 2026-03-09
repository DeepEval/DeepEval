import torch
import numpy as np

def convolve(signal, kernel, mode="full"):
    signal_fft = torch.fft.fft(signal, dim=-1)
    kernel_fft = torch.fft.fft(kernel, dim=-1)
    output_fft = signal_fft * kernel_fft
    output = torch.fft.ifft(output_fft, dim=-1).real
    
    if mode == 'full':
      return output
    elif mode == 'valid':
      padding_length = max(signal.size(-1), kernel.size(-1)) - min(signal.size(-1), kernel.size(-1)) + 1
      return output[:, :padding_length]
    elif mode == 'same':
      padding_length = max(signal.size(-1), kernel.size(-1))
      padding = (padding_length - signal.size(-1)) // 2
      return torch.nn.functional.pad(output, (padding, padding))
    else:
      raise ValueError("Invalid mode. Choose from: 'full', 'valid', 'same'.")

if __name__ == "__main__":
    signal = torch.randn(2, 5)
    kernel = torch.randn(2, 3)
    
    print("Signal:")
    print(signal)
    print("\nKernel:")
    print(kernel)
    
    full_output = convolve(signal, kernel, mode='full')
    print("\nFull Convolution Output:")
    print(full_output)
    
    valid_output = convolve(signal, kernel, mode='valid')
    print("\nValid Convolution Output:")
    print(valid_output)
    
    same_output = convolve(signal, kernel, mode='same')
    print("\nSame Convolution Output:")
    print(same_output)