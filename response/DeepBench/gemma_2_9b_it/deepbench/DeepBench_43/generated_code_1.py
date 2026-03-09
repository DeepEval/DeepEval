import tensorflow as tf

@tf.function
def check_shapes(shapes):
  shapes = list(shapes.split(","))
  for shape in shapes:
    shape = shape.strip()
    if shape:
      dimensions = shape.split(":")
      if len(dimensions) != 2:
        raise ValueError(f"Invalid shape string: {shape}")
      arg_name, expected_shape = dimensions
      try:
        tf.shape(tf.zeros(expected_shape)).numpy()
      except Exception as e:
        raise ValueError(f"Invalid shape string: {shape} due to {e}")

@check_shapes( "x: [D, broadcast N]", "mu: [D, broadcast N]", "L: [D, D]", "return: [N]", )
def multivariate_normal(x: tf.Tensor, mu: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
  
  log_det_L = tf.linalg.logdet(L)
  sqrt_det_L = tf.math.exp(0.5 * log_det_L)
  
  return -0.5 * tf.reduce_sum((x - mu)**2 / tf.linalg.diag_part(tf.matmul(L, L, transpose_a=True)) , axis=0) - 0.5 * log_det_L - 0.5 * tf.cast(tf.shape(x)[1], tf.float32) * tf.math.log(2 * tf.math.pi)

if __name__ == "__main__":
    x = tf.random.normal(shape=[2, 3])
    mu = tf.constant([1.0, 2.0, 3.0])
    L = tf.linalg.cholesky(tf.random.normal(shape=[3, 3]))
    
    log_density = multivariate_normal(x, mu, L)
    print(log_density)