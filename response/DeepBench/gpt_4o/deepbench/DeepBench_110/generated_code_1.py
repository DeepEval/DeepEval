import numpy as np

def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    if padding not in ["same", "valid", "full"]:
        raise ValueError(f"Unknown padding type: {padding}")

    input_length = input_shape[0]
    output_length = output_shape[0]
    kernel_size = kernel_size[0]
    stride = stride[0]

    if padding == "same":
        expected_output_length = (input_length - 1) * stride + 1
    elif padding == "valid":
        expected_output_length = (input_length - 1) * stride + kernel_size
    elif padding == "full":
        expected_output_length = (input_length - 1) * stride + kernel_size + (kernel_size - 1)

    output_padding = output_length - expected_output_length
    return output_padding

if __name__ == "__main__":
    input_shape = (5,)
    output_shape = (15,)
    kernel_size = (3,)
    stride = (2,)
    padding = "same"
    
    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding)
    print(f"Output Padding for 'same': {output_padding}")
    
    padding = "valid"
    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding)
    print(f"Output Padding for 'valid': {output_padding}")
    
    padding = "full"
    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding)
    print(f"Output Padding for 'full': {output_padding}")