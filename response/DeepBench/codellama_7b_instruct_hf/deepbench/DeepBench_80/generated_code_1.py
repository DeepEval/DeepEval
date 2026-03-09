import torch
import torch.nn.functional as F

def convolve(signal, kernel, mode="full"):
    """
    Computes the 1-d convolution of signal by kernel using FFTs.
    The two arguments should have the same rightmost dim, but may otherwise be arbitrarily broadcastable.
    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: A tensor with broadcasted shape.
    """
    if mode not in ["full", "valid", "same"]:
        raise ValueError("Invalid mode: {}. Must be one of 'full', 'valid', or 'same'".format(mode))

    # Compute the FFTs of the signal and kernel
    signal_fft = F.fft(signal, dim=-1)
    kernel_fft = F.fft(kernel, dim=-1)

    # Compute the convolution using the FFT
    conv_fft = signal_fft * kernel_fft

    # Compute the IFFT
    conv = F.ifft(conv_fft, dim=-1)

    # Truncate the convolution to the desired size
    if mode == "full":
        # Keep the full convolution
        pass
    elif mode == "valid":
        # Truncate the convolution to the valid region
        conv = conv[:, :, :signal.size(-1)]
    elif mode == "same":
        # Truncate the convolution to the same size as the input signal
        conv = conv[:, :, :signal.size(-1) + kernel.size(-1) - 1]

    return conv

if __name__ == "__main__":
    # Create sample input values
    signal = torch.randn(1, 3, 5)
    kernel = torch.randn(1, 3, 3)

    # Compute the convolution
    conv = convolve(signal, kernel, mode="full")

    # Print the results
    print(conv)