import torch

def convolve(signal, kernel, mode="full"):
    # Perform 1-D convolution using FFTs
    signal_fft = torch.fft.fft(signal)
    kernel_fft = torch.fft.fft(kernel)
    convolved_fft = signal_fft * kernel_fft

    # Inverse FFT to get the convolved signal
    convolved_signal = torch.fft.ifft(convolved_fft)

    # Apply convolution mode
    if mode == "full":
        # Output shape: m + n - 1
        output = convolved_signal
    elif mode == "valid":
        # Output shape: max(m, n) - min(m, n) + 1
        output = convolved_signal[min(signal.size(-1), kernel.size(-1)) - 1:max(signal.size(-1), kernel.size(-1))]
    elif mode == "same":
        # Output shape: max(m, n)
        output = convolved_signal[max(signal.size(-1), kernel.size(-1)) - max(signal.size(-1), kernel.size(-1)):]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return output

if __name__ == "__main__":
    # Sample input values
    signal = torch.tensor([1, 2, 3, 4, 5])
    kernel = torch.tensor([2, 3, 4])

    # Call the function and print the results
    print(convolve(signal, kernel, mode="full"))
    print(convolve(signal, kernel, mode="valid"))
    print(convolve(signal, kernel, mode="same"))