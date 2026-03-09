import numpy as np

def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    """Calculates the output_padding for a 1D Conv3DTranspose operation.

    Args:
        input_shape: The shape of the input tensor.
        output_shape: The desired shape of the output tensor.
        kernel_size: The size of the kernel.
        stride: The stride of the convolution.
        padding: The padding type.

    Returns:
        The calculated output_padding.

    Raises:
        ValueError: If an unknown padding type is provided.
    """

    if padding not in ["same", "valid", "full"]:
        raise ValueError(f"Unknown padding type: {padding}")

    input_size = input_shape[-1]
    output_size = output_shape[-1]
    kernel_size = kernel_size[-1]
    stride = stride[-1]

    if padding == "same":
        output_padding = (kernel_size - stride) // 2
    elif padding == "valid":
        output_padding = 0
    elif padding == "full":
        output_padding = kernel_size - stride

    return output_padding


if __name__ == "__main__":
    # Sample input values
    input_shape = (1, 1, 10)
    output_shape = (1, 1, 20)
    kernel_size = (1, 1, 3)
    stride = (1, 1, 2)
    padding = "same"

    # Call the function
    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding)

    # Print the results
    print(f"Output padding: {output_padding}")