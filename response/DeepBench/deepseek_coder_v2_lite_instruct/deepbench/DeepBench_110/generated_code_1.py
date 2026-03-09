def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    import numpy as np

    if padding == "same":
        output_padding = (input_shape - 1) * stride - output_shape + kernel_size
    elif padding == "valid":
        output_padding = (input_shape - 1) * stride - output_shape + 1
    elif padding == "full":
        output_padding = (input_shape - 1) * stride - output_shape + kernel_size
    else:
        raise ValueError("Unknown padding type")

    return output_padding

if __name__ == "__main__":
    input_shape = 5
    output_shape = 7
    kernel_size = 3
    stride = 2

    print("Same Padding Output Padding:", _deconv_output_padding(input_shape, output_shape, kernel_size, stride, "same"))
    print("Valid Padding Output Padding:", _deconv_output_padding(input_shape, output_shape, kernel_size, stride, "valid"))
    print("Full Padding Output Padding:", _deconv_output_padding(input_shape, output_shape, kernel_size, stride, "full"))