import torch
import torch.nn.functional as F

def convolve(signal, kernel, mode="full"):
    signal_size = signal.size(-1)
    kernel_size = kernel.size(-1)

    # Compute the FFT of the signal and kernel
    signal_fft = torch.fft.rfft(signal)
    kernel_fft = torch.fft.rfft(kernel)

    # Perform element-wise multiplication in the frequency domain
    convolved_fft = signal_fft * kernel_fft

    # Compute the inverse FFT to get back to the time domain
    convolved = torch.fft.irfft(convolved_fft)

    # Determine output shape based on the mode
    if mode == "full":
        return convolved
    elif mode == "valid":
        return convolved[(kernel_size - 1):-(kernel_size - 1)]
    elif mode == "same":
        if signal_size >= kernel_size:
            start = (signal_size - kernel_size) // 2
            return convolved[start:start + signal_size]
        else:
            start = (kernel_size - signal_size) // 2
            return convolved[start:start + kernel_size]
    else:
        raise ValueError("Invalid mode. Choose from 'full', 'valid', or 'same'.")

if __name__ == "__main__":
    signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    kernel = torch.tensor([0.2, 0.5, 0.3])
    
    result_full = convolve(signal, kernel, mode="full")
    result_valid = convolve(signal, kernel, mode="valid")
    result_same = convolve(signal, kernel, mode="same")

    print("Full Convolution Result:", result_full)
    print("Valid Convolution Result:", result_valid)
    print("Same Convolution Result:", result_same)