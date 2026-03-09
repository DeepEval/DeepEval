import tensorflow as tf

def do_conv2d(conv, input_, padding='SAME', weight=None, bias=None):
  if weight is None:
    weight = conv.kernel
  if bias is None:
    bias = conv.bias
  return tf.nn.conv2d(input_, weight, strides=conv.strides, padding=padding) + bias 

if __name__ == "__main__":
  input_ = tf.random.normal(shape=(1, 10, 10, 3))
  conv = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1))
  output = do_conv2d(conv, input_)
  print(output.shape)