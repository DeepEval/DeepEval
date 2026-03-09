import torch
from torch.nn import Conv3DTranspose

def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    if padding == "same":
        output_padding = (kernel_size - 1) // 2
    elif padding == "valid":
        output_padding = 0
    elif padding == "full":
        output_padding = kernel_size - 1
    else:
        raise ValueError(f"Unknown padding type: {padding}")

    return output_padding

if __name__ == "__main__":
    # Sample input values
    input_shape = (1, 3, 3, 3)
    output_shape = (1, 3, 3, 3)
    kernel_size = 2
    stride = 1
    padding = "same"

    # Call function and print output
    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding)
    print(f"Output padding: {output_padding}")