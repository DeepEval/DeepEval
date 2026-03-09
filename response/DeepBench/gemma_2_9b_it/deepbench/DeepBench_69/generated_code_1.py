import tensorflow as tf

def filter3d(input, kernel, border_type, normalized):
  if normalized:
    kernel = kernel / tf.reduce_sum(kernel)
  padding = {'data_format': 'NDHWC', 'mode': border_type}
  return tf.nn.conv3d(input, kernel, strides=[1, 1, 1, 1, 1], padding=padding)

if __name__ == "__main__":
  input_tensor = tf.random.normal(shape=(1, 3, 10, 10, 10))
  kernel = tf.random.normal(shape=(3, 3, 3, 1, 1))
  output = filter3d(input_tensor, kernel, 'SAME', True)
  print(output.shape)