import torch

def convolve(signal, kernel, mode="full"):
    # Compute the FFTs of the signal and kernel
    signal_fft = torch.fft.rfftn(signal)
    kernel_fft = torch.fft.rfftn(kernel)
    
    # Compute the element-wise product of the FFTs
    convolution_fft = signal_fft * kernel_fft
    
    # Compute the inverse FFT to get the convolution result
    convolution = torch.fft.irfftn(convolution_fft, s=signal.size(-1) + kernel.size(-1) - 1)
    
    # Adjust the size based on the mode
    if mode == "full":
        return convolution
    elif mode == "valid":
        m, n = signal.size(-1), kernel.size(-1)
        return convolution[..., :max(m, n) - min(m, n) + 1]
    elif mode == "same":
        m, n = signal.size(-1), kernel.size(-1)
        return convolution[..., (m - 1) // 2:(m - 1) // 2 + max(m, n)]

if __name__ == "__main__":
    # Create sample input values
    signal = torch.tensor([1.0, 2.0, 3.0, 4.0])
    kernel = torch.tensor([1.0, 1.0, 1.0])
    
    # Call the function and print the results
    result_full = convolve(signal, kernel, mode="full")
    result_valid = convolve(signal, kernel, mode="valid")
    result_same = convolve(signal, kernel, mode="same")
    
    print("Full mode result:", result_full)
    print("Valid mode result:", result_valid)
    print("Same mode result:", result_same)