import torch
import numpy as np

def convolve(signal, kernel, mode="full"):
    # Convert kernel to torch Tensor
    kernel_torch = torch.from_numpy(kernel)
    # Ensure signal and kernel have the same rightmost dimension
    signal = signal.reshape(-1, 1)
    kernel = kernel.reshape(1, -1)
    # Compute the 1-d convolution using FFTs
    convolved = torch.fft.fft(signal * kernel_torch, dim=-1)
    # Remove the conjugate of the FFT
    convolved = torch.fft.ifft(convolved).real
    # Apply inverse reshape
    convolved = convolved.reshape(signal.shape[:-1] + kernel.shape[::-1])
    # Perform convolution according to the specified mode
    if mode == "full":
        rightmost_size = convolved.size(-1) + kernel.size(-1) - 1
    elif mode == "valid":
        rightmost_size = max(convolved.size(-1), kernel.size(-1)) - min(convolved.size(-1), kernel.size(-1)) + 1
    else:  # mode == "same"
        rightmost_size = max(convolved.size(-1), kernel.size(-1))
    return convolved.resize_(convolved.shape[:-1] + (rightmost_size,))

if __name__ == "__main__":
    # Sample input values
    signal = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    kernel = torch.tensor([0.5, 1, 0.5], dtype=torch.float32)

    # Call the function and print the results
    result = convolve(signal, kernel, mode="full")
    print("Convolution (full):", result.numpy())

    result = convolve(signal, kernel, mode="valid")
    print("Convolution (valid):", result.numpy())

    result = convolve(signal, kernel, mode="same")
    print("Convolution (same):", result.numpy())