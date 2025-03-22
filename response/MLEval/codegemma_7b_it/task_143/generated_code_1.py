import tensorflow as tf

def method():
  x = tf.constant([10, 20, 30, 40, 50])
  y = tf.constant([3, 5, 7, 9, 11])

  output = tf.cast(tf.math.floormod(x, y), dtype=tf.int32)

  return output

print(method())