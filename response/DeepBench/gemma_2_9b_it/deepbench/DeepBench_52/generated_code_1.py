import tensorflow as tf

def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
  kernel = tf.expand_dims(kernel, axis=[1, 2, 3])
  return tf.nn.conv3d(tensor, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')

if __name__ == "__main__":
  tensor = tf.random.normal(shape=(1, 8, 8, 8, 1))
  kernel = tf.random.normal(shape=(1,))
  result = separable_filter(tensor, kernel)
  print(result.shape)