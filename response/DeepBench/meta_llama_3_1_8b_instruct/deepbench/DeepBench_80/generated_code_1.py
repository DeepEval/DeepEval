import torch
import torch.fft

def convolve(signal, kernel, mode="full"):
    # Calculate the FFT of the signal and the kernel
    signal_fft = torch.fft.fft(signal, dim=-1)
    kernel_fft = torch.fft.fft(kernel, dim=-1)

    # Calculate the FFT of the product of the signal and the kernel
    product_fft = signal_fft * kernel_fft

    # Calculate the inverse FFT of the product
    result_fft = torch.fft.ifft(product_fft, dim=-1)

    # Apply the specified mode to the result
    if mode == "full":
        result = torch.real(result_fft)
    elif mode == "valid":
        signal_size = signal.size(-1)
        kernel_size = kernel.size(-1)
        result = torch.real(result_fft[:, :, :-kernel_size + 1])
    elif mode == "same":
        signal_size = signal.size(-1)
        kernel_size = kernel.size(-1)
        padding = (kernel_size - 1) // 2
        result = torch.real(result_fft)[:, :, padding:-padding]
    else:
        raise ValueError("Invalid mode")

    return result

if __name__ == "__main__":
    # Create sample input values
    signal = torch.randn(1, 10)
    kernel = torch.randn(1, 3)

    # Call the function
    result_full = convolve(signal, kernel, mode="full")
    result_valid = convolve(signal, kernel, mode="valid")
    result_same = convolve(signal, kernel, mode="same")

    # Print the results
    print("Full mode result:")
    print(result_full)
    print("\nValid mode result:")
    print(result_valid)
    print("\nSame mode result:")
    print(result_same)