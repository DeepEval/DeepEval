import numpy as np

def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    if padding not in ['same', 'valid', 'full']:
        raise ValueError("Unknown padding type. Supported types are: 'same', 'valid', 'full'.")

    input_length = input_shape[-1]
    output_length = output_shape[-1]

    if padding == 'valid':
        output_length_calculated = (input_length - 1) * stride + kernel_size
    elif padding == 'same':
        output_length_calculated = input_length * stride
    elif padding == 'full':
        output_length_calculated = (input_length - 1) * stride + kernel_size - 1

    output_padding = output_length - output_length_calculated

    if output_padding < 0:
        output_padding = 0

    return output_padding

if __name__ == "__main__":
    input_shape = (1, 1, 10)  # Example input shape (batch_size, channels, length)
    output_shape = (1, 1, 12)  # Example output shape
    kernel_size = 3
    stride = 2
    padding = 'same'
    
    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding)
    print("Calculated output padding:", output_padding)