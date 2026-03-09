import tensorflow as tf

def _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding_type):
  if padding_type == "same":
    dilated_kernel_size = kernel_size
    if isinstance(dilated_kernel_size, int):
      dilated_kernel_size = [dilated_kernel_size] * len(kernel_size)
    padding = tf.math.ceil((output_shape - input_shape + dilated_kernel_size - 1) / stride) - input_shape
    output_padding = [(p - 1) * stride for p in padding]
  elif padding_type == "valid":
    output_padding = [0] * len(output_shape)
  elif padding_type == "full":
    padding = ((kernel_size[0] - 1) * stride) // 2
    output_padding = [padding] * len(output_shape)
  else:
    raise ValueError("Unknown padding type: {}".format(padding_type))
  return output_padding

if __name__ == "__main__":
    input_shape = [10, 32, 64]
    output_shape = [16, 32, 128]
    kernel_size = 3
    stride = 2
    padding_type = "same"
    output_padding = _deconv_output_padding(input_shape, output_shape, kernel_size, stride, padding_type)
    print(f"Output padding: {output_padding}")