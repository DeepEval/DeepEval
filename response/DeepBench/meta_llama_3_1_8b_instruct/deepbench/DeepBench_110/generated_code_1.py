import numpy as np

def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    if padding not in ["same", "valid", "full"]:
        raise ValueError("Unknown padding type. Supported padding types are'same', 'valid', and 'full'.")

    output_padding = 0
    if padding == "same":
        output_padding = (kernel_size - 1) * stride - (input_shape[0] - output_shape[0])
    elif padding == "valid":
        output_padding = (kernel_size - 1) * stride
    elif padding == "full":
        output_padding = (kernel_size - 1) * stride + (input_shape[0] - output_shape[0])

    return output_padding

if __name__ == "__main__":
    input_shape = (10, 10, 10)
    output_shape = (15, 15, 15)
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = "same"

    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding)
    print(f"Output padding for {padding} padding type: {output_padding}")