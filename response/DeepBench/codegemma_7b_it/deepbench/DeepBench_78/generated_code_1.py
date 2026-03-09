import torch
import torch.nn as nn

def filter2d(input, kernel, border_type='constant', normalized=False, padding='same', behavior='corr'):
    """Convolve a tensor with a 2d kernel.

    Args:
        input: the input tensor with shape of :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving. The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding. 2 modes available ``'same'`` or ``'valid'``.
        behaviour: defines the convolution mode -- correlation (default), using pytorch conv2d, or true convolution (kernel is flipped). 2 modes available ``'corr'`` or ``'conv'``.

    Return:
        Tensor: the convolved tensor of same size and numbers of channels as the input with shape :math:`(B, C, H, W)`.
    """

    if normalized:
        kernel /= torch.sum(kernel)

    if len(kernel.shape) == 2:
        kernel = kernel.unsqueeze(0)

    if len(kernel.shape) == 3:
        if kernel.shape[0] != input.shape[0]:
            raise ValueError('Invalid kernel shape: the batch dimension must match the input dimension.')

    if padding == 'same':
        input_padding = (kernel.shape[2] - 1) // 2
        input = nn.functional.pad(input, (input_padding, input_padding, input_padding, input_padding), border_type)

    if behavior == 'conv':
        kernel = kernel.flip(2, 3)

    output = nn.functional.conv2d(input, kernel, padding=0, stride=1, groups=input.shape[1])

    return output

if __name__ == "__main__":
    # Example usage:
    input = torch.randn(1, 3, 5, 5)
    kernel = torch.randn(1, 3, 3)
    output = filter2d(input, kernel)
    print(output.shape)