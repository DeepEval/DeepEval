import torch
import torch.fft

def convolve(signal, kernel, mode="full"):
    signal = torch.as_tensor(signal)
    kernel = torch.as_tensor(kernel)

    m = signal.size(-1)
    n = kernel.size(-1)
    fft_size = m + n - 1

    # Perform FFT on both signal and kernel
    signal_f = torch.fft.rfft(signal, fft_size)
    kernel_f = torch.fft.rfft(kernel, fft_size)

    # Element-wise multiplication in frequency domain
    result_f = signal_f * kernel_f

    # Inverse FFT to convert back to time domain
    result = torch.fft.irfft(result_f, fft_size)

    # Slicing the result based on the mode
    if mode == "full":
        return result
    elif mode == "valid":
        valid_size = max(m, n) - min(m, n) + 1
        start_idx = min(m, n) - 1
        end_idx = start_idx + valid_size
        return result[..., start_idx:end_idx]
    elif mode == "same":
        same_size = max(m, n)
        start_idx = (fft_size - same_size) // 2
        end_idx = start_idx + same_size
        return result[..., start_idx:end_idx]
    else:
        raise ValueError("mode must be one of: 'full', 'valid', 'same'")

if __name__ == "__main__":
    # Sample input values
    signal = torch.tensor([1, 2, 3, 4])
    kernel = torch.tensor([0.5, 1, 0.5])

    # Call the function with different modes
    result_full = convolve(signal, kernel, mode="full")
    result_valid = convolve(signal, kernel, mode="valid")
    result_same = convolve(signal, kernel, mode="same")

    # Print the results
    print("Full convolution:", result_full)
    print("Valid convolution:", result_valid)
    print("Same convolution:", result_same)