import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose

def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding):
    if padding.lower() == 'same':
        output_size = output_shape
        input_size = input_shape
        div, mod = divmod(input_shape[-1] - 1, stride[0]) + 1
        return (stride[0] - 1, 0) if mod else (stride[0], 0)
    elif padding.lower() == 'valid':
        return (0, 0)
    elif padding.lower() == 'full':
        return (0, output_shape[-1] - 1)
    else:
        raise ValueError("Unknown padding type")

def _check_input(input_shape, output_shape, kernel_size, stride, padding):
    if not all(isinstance(i, int) for i in input_shape):
        raise ValueError("Input shape must be an integer")
    if not all(isinstance(i, int) for i in output_shape):
        raise ValueError("Output shape must be an integer")
    if not all(isinstance(i, int) for i in kernel_size):
        raise ValueError("Kernel size must be an integer")
    if not all(isinstance(i, int) for i in stride):
        raise ValueError("Stride must be an integer")
    if not isinstance(padding, str):
        raise ValueError("Padding must be a string")

if __name__ == "__main__":
    _check_input(input_shape=(5, 5, 5), output_shape=(10, 10, 10), kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='same')
    output_padding = _deconv_output_padding(input_shape=(5, 5, 5), output_shape=(10, 10, 10), kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='same')
    print(output_padding)